#!/usr/bin/env python3
"""
BF-VAE v2 Fine-tune Training Script
=====================================
Three fixes over v1:
  1. beat_loss direction: 1 - regularity  (v1 was regularity → trained WEAKER beats)
  2. Frequency-split MSE: only apply MSE on bands >= LOW_FREQ_CUTOFF
     → model is free to modify low-freq rhythm while preserving melody
  3. KL weight: default 0.01 (v1 was 0.001, nearly disabled)

Usage (Colab):
  python train_v2.py \
      --data_dir /content/fma_test_audio \
      --checkpoint ../BF_VAE_results/checkpoints_v2/best_model.pth \
      --save_dir   /content/drive/MyDrive/BF_VAE_v2_results \
      --epochs 40 \
      --beat_weight 2.0 \
      --kl_weight   0.01
"""

import sys, os, argparse, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ── resolve project paths ─────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

for p in [
    os.path.join(PROJECT_ROOT, '2.Music_dataset'),
    os.path.join(PROJECT_ROOT, '3.Model'),
    SCRIPT_DIR,
    PROJECT_ROOT,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from audio_dataset_v2 import AudioMelDataset_v2 as MusicDataset
from vae_model        import MelSpectrogramVAE as VAE
from beat_loss_v2     import beat_loss_v2, compute_regularity_score


# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='BF-VAE v2 Fine-tune')
    p.add_argument('--data_dir',    required=True,  help='Weak-beat audio folder')
    p.add_argument('--checkpoint',  default=None,   help='Path to v1 .pth checkpoint (fine-tune)')
    p.add_argument('--save_dir',    default='./checkpoints_v2', help='Output folder')
    p.add_argument('--epochs',      type=int,   default=40)
    p.add_argument('--batch_size',  type=int,   default=16)
    p.add_argument('--lr',          type=float, default=5e-5,
                   help='Use lower LR for fine-tuning (5e-5); from scratch: 1e-4')
    p.add_argument('--latent_dim',  type=int,   default=128)
    p.add_argument('--kl_weight',   type=float, default=0.01,
                   help='KL weight (was 0.001 in v1; 0.01 gives gentle regularisation)')
    p.add_argument('--beat_weight', type=float, default=2.0,
                   help='Beat loss weight (higher = stronger beat push; recommend 1.5-3.0)')
    p.add_argument('--warmup_epochs', type=int, default=10,
                   help='Epochs to ramp beat_weight from 0 to target')
    p.add_argument('--low_freq_cutoff', type=int, default=16,
                   help='MSE applied only to mel bands >= this (frees low-freq for beat)')
    p.add_argument('--log_interval', type=int, default=5, help='Print every N epochs')
    p.add_argument('--train_ratio', type=float, default=0.70, help='Train split ratio')
    p.add_argument('--val_ratio',   type=float, default=0.15, help='Val split ratio')
    # test_ratio = 1 - train_ratio - val_ratio  (default 0.15)
    p.add_argument('--split_seed',  type=int,   default=42,   help='Random seed for split')
    p.add_argument('--split_file',  type=str,   default=None,
                   help='If given, load/save the train/val/test file split as JSON '
                        '(ensures test set never leaks into training across runs)')
    return p.parse_args()


# ── loss computation ──────────────────────────────────────────────────────────
FREE_BITS = 0.5   # minimum KL nats per latent dim (anti-collapse)

def compute_losses(recon, target, mu, logvar,
                   kl_w, beat_w, low_freq_cutoff):
    """
    Frequency-split reconstruction loss + KL (with free bits) + beat loss v2.

    Anti-collapse: free bits ensures each latent dim contributes at least
    FREE_BITS nats of KL, preventing posterior collapse even with low kl_w.

    recon/target: (B, 1, 128, 431)
    Returns: total_loss, dict of components
    """
    # High-freq MSE: preserve melody/timbre (bands >= low_freq_cutoff)
    recon_high  = recon[:, :, low_freq_cutoff:, :]
    target_high = target[:, :, low_freq_cutoff:, :]
    recon_loss  = nn.functional.mse_loss(recon_high, target_high)

    # Small all-freq anchor so low-freq doesn't go completely wild
    recon_loss_all = nn.functional.mse_loss(recon, target)
    recon_loss = 0.7 * recon_loss + 0.3 * recon_loss_all

    # KL divergence per latent dimension
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, latent_dim)
    # Free bits: clamp minimum KL to FREE_BITS per dim (prevents collapse)
    kl_per_dim = torch.clamp(kl_per_dim.mean(dim=0), min=FREE_BITS)
    kl_loss = kl_per_dim.sum()

    # Beat loss v2: 1 - regularity → minimise = STRONGER beats
    b_loss = beat_loss_v2(recon)

    total = recon_loss + kl_w * kl_loss + beat_w * b_loss

    # Collapse indicator: if mean KL << FREE_BITS, decoder is ignoring z
    kl_raw = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean().item()

    return total, {
        'total':      total.item(),
        'recon':      recon_loss.item(),
        'kl':         kl_loss.item(),
        'kl_raw':     kl_raw,           # for collapse monitoring
        'beat':       b_loss.item(),
        'regularity': 1.0 - b_loss.item(),
    }


# ── training / validation loops ───────────────────────────────────────────────
def run_epoch(model, loader, optimizer, device, kl_w, beat_w,
              low_freq_cutoff, train=True):
    model.train(train)
    totals = {'total': 0, 'recon': 0, 'kl': 0, 'beat': 0, 'regularity': 0}
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            if recon.size(-1) != batch.size(-1):
                recon = recon[:, :, :, :batch.size(-1)]

            loss, info = compute_losses(recon, batch, mu, logvar,
                                        kl_w, beat_w, low_freq_cutoff)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = batch.size(0)
            for k in totals:
                totals[k] += info[k] * bs
            n += bs

    return {k: v / n for k, v in totals.items()}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')   # Apple Silicon GPU
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')
    os.makedirs(args.save_dir, exist_ok=True)

    # ── dataset & 3-way split (by FILE, not by window) ────────────────────
    print('Loading dataset...')
    dataset = MusicDataset(data_dir=args.data_dir)

    split_path = args.split_file or os.path.join(args.save_dir, 'data_split.json')

    if os.path.exists(split_path):
        # Load existing split → guarantees identical test set across re-runs
        with open(split_path) as f:
            split = json.load(f)
        train_files = set(split['train'])
        val_files   = set(split['val'])
        test_files  = set(split['test'])
        print(f'Loaded existing split from {split_path}')
    else:
        # Build new split by file (not by window index)
        all_files = sorted(set(str(fp) for fp, _ in dataset.samples))
        random.seed(args.split_seed)
        random.shuffle(all_files)

        n  = len(all_files)
        n_train = int(args.train_ratio * n)
        n_val   = int(args.val_ratio   * n)
        # remaining goes to test
        train_files = set(all_files[:n_train])
        val_files   = set(all_files[n_train:n_train + n_val])
        test_files  = set(all_files[n_train + n_val:])

        split = {
            'train': sorted(train_files),
            'val':   sorted(val_files),
            'test':  sorted(test_files),
            'seed':  args.split_seed,
            'ratios': [args.train_ratio, args.val_ratio,
                       round(1 - args.train_ratio - args.val_ratio, 4)],
        }
        os.makedirs(args.save_dir, exist_ok=True)
        with open(split_path, 'w') as f:
            json.dump(split, f, indent=2)
        print(f'Created new split → saved to {split_path}')

    # Map window indices to splits
    train_idx = [i for i, (fp, _) in enumerate(dataset.samples) if str(fp) in train_files]
    val_idx   = [i for i, (fp, _) in enumerate(dataset.samples) if str(fp) in val_files]
    test_idx  = [i for i, (fp, _) in enumerate(dataset.samples) if str(fp) in test_files]

    print(f'\nSplit (by file):')
    print(f'  Train : {len(train_files):4d} files  → {len(train_idx):5d} windows')
    print(f'  Val   : {len(val_files):4d} files  → {len(val_idx):5d} windows')
    print(f'  Test  : {len(test_files):4d} files  → {len(test_idx):5d} windows')
    print(f'  NOTE: Test set is saved to {split_path} and will NOT be used during training.')

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    # test_ds is intentionally NOT used here — use evaluate_v2.py after training

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── model ─────────────────────────────────────────────────────────────
    model = VAE(latent_dim=args.latent_dim).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        print(f'Loaded checkpoint: {args.checkpoint}')
    else:
        print('No checkpoint found – training from scratch.')

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model parameters: {params:.1f}M')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True)

    # ── training loop ─────────────────────────────────────────────────────
    history     = []
    best_reg    = -1.0
    best_path   = os.path.join(args.save_dir, 'best_model_v2.pth')
    latest_path = os.path.join(args.save_dir, 'latest_model_v2.pth')

    for epoch in range(1, args.epochs + 1):
        # Warm-up: linearly ramp beat_weight from 0 to target
        if epoch <= args.warmup_epochs:
            beat_w = args.beat_weight * epoch / args.warmup_epochs
        else:
            beat_w = args.beat_weight

        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, device,
                            args.kl_weight, beat_w, args.low_freq_cutoff, train=True)
        val_m   = run_epoch(model, val_loader,   optimizer, device,
                            args.kl_weight, beat_w, args.low_freq_cutoff, train=False)
        scheduler.step(val_m['total'])

        history.append({'epoch': epoch, 'train': train_m, 'val': val_m,
                        'beat_w': beat_w, 'lr': optimizer.param_groups[0]['lr']})

        # Always save latest
        torch.save({'model_state_dict': model.state_dict(),
                    'epoch': epoch, 'val_regularity': val_m['regularity']},
                   latest_path)

        # Save best (highest beat regularity on val set)
        if val_m['regularity'] > best_reg:
            best_reg = val_m['regularity']
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch, 'val_regularity': best_reg},
                       best_path)
            star = ' ★ BEST'
        else:
            star = ''

        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            elapsed = time.time() - t0
            # Collapse warning: if kl_raw < 0.1 nats avg, latent code is being ignored
            kl_warn = ' ⚠ KL COLLAPSE?' if val_m['kl_raw'] < 0.1 else ''
            print(f'Epoch {epoch:3d}/{args.epochs} '
                  f'| train reg={train_m["regularity"]:.3f} '
                  f'| val reg={val_m["regularity"]:.3f} beat={val_m["beat"]:.4f} '
                  f'kl_raw={val_m["kl_raw"]:.3f} '
                  f'| beat_w={beat_w:.2f} | {elapsed:.0f}s{star}{kl_warn}')

    # Save training history
    with open(os.path.join(args.save_dir, 'history_v2.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\nDone!  Best val regularity: {best_reg:.4f}')
    print(f'Best model saved to: {best_path}')


if __name__ == '__main__':
    main()
