#!/usr/bin/env python3
"""
BF-VAE v2  ·  Final Evaluation on Test Set
============================================
ONLY run this AFTER training is complete.
Reads the test file list from data_split.json (saved by train_v2.py)
so the test set is guaranteed to be files the model NEVER saw during training.

Usage:
    python evaluate_v2.py \
        --checkpoint  7.BF_VAE_v2/checkpoints/best_model_v2.pth \
        --split_file  7.BF_VAE_v2/checkpoints/data_split.json \
        --data_dir    ./weak_beat_music \
        --output_dir  7.BF_VAE_v2/test_results \
        --n_samples   20
"""

import sys, os, argparse, json, random
import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for p in [os.path.join(PROJECT_ROOT, '3.Model'), SCRIPT_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vae_model      import MelSpectrogramVAE as VAE
from beat_loss_v2   import compute_regularity_score
from inference_v2   import (audio_to_mel, mel_to_audio, load_full_audio,
                             chunk_audio, overlap_add, detect_beat_strength,
                             WEAK_BEAT_THRESHOLD, SR, HOP_LENGTH, N_FRAMES, NORM_DIV)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='BF-VAE v2 Test-set Evaluation')
    p.add_argument('--checkpoint',  required=True,  help='Path to best_model_v2.pth')
    p.add_argument('--split_file',  required=True,
                   help='Path to data_split.json (created by train_v2.py)')
    p.add_argument('--data_dir',    required=True,  help='Root audio folder')
    p.add_argument('--output_dir',  default='./test_results_v2')
    p.add_argument('--n_samples',   type=int, default=20,
                   help='How many test files to evaluate (0 = all)')
    p.add_argument('--save_audio',  action='store_true',
                   help='Also save enhanced .wav for each sample')
    p.add_argument('--latent_dim',  type=int, default=128)
    p.add_argument('--device',      default='auto')
    return p.parse_args()


# ── device helper ─────────────────────────────────────────────────────────────
def get_device(device_str):
    if device_str == 'auto':
        if torch.cuda.is_available():   return torch.device('cuda')
        if torch.backends.mps.is_available(): return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_str)


# ── single-file evaluation ─────────────────────────────────────────────────────
def evaluate_file(fp, model, device):
    """
    Process one file: detect → enhance → compute metrics.
    Returns a result dict.
    """
    try:
        y_in = load_full_audio(fp)
    except Exception as e:
        return None

    dur = len(y_in) / SR

    # Detect BEFORE enhancement
    detection = detect_beat_strength(y_in)

    # Chunk → VAE → overlap-add
    chunks_in  = chunk_audio(y_in)
    chunks_out = []
    reg_out_scores = []

    with torch.no_grad():
        for chunk, start, end in chunks_in:
            mel_in = audio_to_mel(chunk)
            t_in   = torch.FloatTensor(mel_in).unsqueeze(0).unsqueeze(0).to(device)
            t_out, _, _ = model(t_in)
            if t_out.shape[-1] != N_FRAMES:
                t_out = t_out[:, :, :, :N_FRAMES]
            mel_out = t_out.squeeze().cpu().numpy()
            audio_out = mel_to_audio(mel_out)
            reg_out_scores.append(compute_regularity_score(t_out.cpu()))
            chunks_out.append((audio_out, start, end))

    y_out   = overlap_add(chunks_out, len(y_in))
    reg_in  = detection['mean_regularity']
    reg_out = float(np.mean(reg_out_scores))

    try:
        bpm_in,  _ = librosa.beat.beat_track(y=y_in,  sr=SR, hop_length=HOP_LENGTH)
        bpm_out, _ = librosa.beat.beat_track(y=y_out, sr=SR, hop_length=HOP_LENGTH)
    except Exception:
        bpm_in = bpm_out = 0.0

    mel_in_ref  = audio_to_mel(y_in[:int(10 * SR)])
    mel_out_ref = audio_to_mel(y_out[:int(10 * SR)])
    mse = float(np.mean((mel_in_ref - mel_out_ref) ** 2))

    return {
        'file':       os.path.basename(fp),
        'duration_s': round(dur, 1),
        'reg_in':     round(reg_in, 4),
        'reg_out':    round(reg_out, 4),
        'reg_delta':  round(reg_out - reg_in, 4),
        'bpm_in':     round(float(bpm_in), 1),
        'bpm_out':    round(float(bpm_out), 1),
        'mse':        round(mse, 6),
        'is_weak_beat_in': detection['is_weak_beat'],
        'improved':   reg_out > reg_in,
        'audio_out':  y_out,   # kept for optional save
    }


# ── summary plot ──────────────────────────────────────────────────────────────
def plot_summary(results, save_path):
    reg_in  = [r['reg_in']    for r in results]
    reg_out = [r['reg_out']   for r in results]
    deltas  = [r['reg_delta'] for r in results]
    mses    = [r['mse']       for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Scatter: reg_in vs reg_out
    ax = axes[0]
    ax.scatter(reg_in, reg_out, c=deltas, cmap='RdYlGn', s=60, edgecolors='k', linewidths=0.5)
    lim = [0, 1]
    ax.plot(lim, lim, 'k--', linewidth=1, label='No change')
    ax.set_xlabel('Regularity BEFORE enhancement')
    ax.set_ylabel('Regularity AFTER enhancement')
    ax.set_title('Beat Regularity: Before vs After')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Histogram of deltas
    ax = axes[1]
    ax.hist(deltas, bins=15, color='steelblue', edgecolor='k', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No change')
    ax.axvline(np.mean(deltas), color='green', linestyle='-', linewidth=1.5,
               label=f'Mean Δ = {np.mean(deltas):+.3f}')
    ax.set_xlabel('Δ Beat Regularity (out − in)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Regularity Improvement')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # MSE distribution
    ax = axes[2]
    ax.hist(mses, bins=15, color='tomato', edgecolor='k', alpha=0.8)
    ax.axvline(np.mean(mses), color='navy', linestyle='-', linewidth=1.5,
               label=f'Mean MSE = {np.mean(mses):.4f}')
    ax.set_xlabel('Reconstruction MSE (10s window)')
    ax.set_ylabel('Count')
    ax.set_title('Reconstruction Quality Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = get_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────
    print(f'Loading model from {args.checkpoint}...')
    model = VAE(latent_dim=args.latent_dim).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f'Device: {device}')

    # ── load test file list ───────────────────────────────────────────────
    print(f'\nReading split from {args.split_file}...')
    with open(args.split_file) as f:
        split = json.load(f)

    test_files = split['test']
    print(f'Test set: {len(test_files)} files  '
          f'(train={len(split["train"])}, val={len(split["val"])}, test={len(test_files)})')
    print(f'Split ratios: {split.get("ratios", "?")}  seed={split.get("seed", "?")}')

    # Resolve absolute paths
    resolved = []
    for fp in test_files:
        if os.path.exists(fp):
            resolved.append(fp)
        else:
            # Try relative to data_dir
            candidate = os.path.join(args.data_dir, os.path.basename(fp))
            if os.path.exists(candidate):
                resolved.append(candidate)

    print(f'Resolved {len(resolved)}/{len(test_files)} test files')
    if not resolved:
        print('ERROR: No test files found. Check --data_dir path.')
        return

    # Optionally subsample
    if args.n_samples > 0 and args.n_samples < len(resolved):
        random.seed(0)
        random.shuffle(resolved)
        resolved = resolved[:args.n_samples]
        print(f'Using {len(resolved)} samples (--n_samples)')

    # ── evaluate ──────────────────────────────────────────────────────────
    print(f'\nEvaluating {len(resolved)} test files...\n')
    header = f"{'File':<30} {'Dur':>5} {'RegIn':>6} {'RegOut':>7} {'Delta':>7} {'BPMIn':>6} {'BPMOut':>7} {'MSE':>8}"
    print(header)
    print('─' * len(header))

    results = []
    for i, fp in enumerate(resolved):
        r = evaluate_file(fp, model, device)
        if r is None:
            print(f'  SKIP: {os.path.basename(fp)} (load error)')
            continue
        results.append(r)

        print(f"{r['file'][:30]:<30} {r['duration_s']:>5.1f} "
              f"{r['reg_in']:>6.3f} {r['reg_out']:>7.3f} "
              f"{r['reg_delta']:>+7.3f} {r['bpm_in']:>6.1f} "
              f"{r['bpm_out']:>7.1f} {r['mse']:>8.5f}")

        if args.save_audio:
            out_wav = os.path.join(args.output_dir,
                                   os.path.splitext(r['file'])[0] + '_enhanced.wav')
            sf.write(out_wav, r['audio_out'], SR)

    # ── summary stats ─────────────────────────────────────────────────────
    print('\n' + '═' * 60)
    reg_in_arr  = np.array([r['reg_in']  for r in results])
    reg_out_arr = np.array([r['reg_out'] for r in results])
    delta_arr   = np.array([r['reg_delta'] for r in results])
    mse_arr     = np.array([r['mse'] for r in results])
    improved    = np.sum([r['improved'] for r in results])

    print(f'  Test samples evaluated  : {len(results)}')
    print(f'  Weak-beat input (< {WEAK_BEAT_THRESHOLD}) : '
          f'{sum(r["is_weak_beat_in"] for r in results)}/{len(results)}')
    print(f'  Regularity BEFORE       : {reg_in_arr.mean():.3f} ± {reg_in_arr.std():.3f}')
    print(f'  Regularity AFTER        : {reg_out_arr.mean():.3f} ± {reg_out_arr.std():.3f}')
    print(f'  Mean Δ Regularity       : {delta_arr.mean():+.3f} ± {delta_arr.std():.3f}')
    print(f'  % Improved              : {improved}/{len(results)} = {improved/len(results)*100:.0f}%')
    print(f'  Mean MSE                : {mse_arr.mean():.4f} ± {mse_arr.std():.4f}')
    print('═' * 60)

    # ── save JSON summary ─────────────────────────────────────────────────
    summary = {
        'checkpoint': args.checkpoint,
        'n_evaluated': len(results),
        'split_ratios': split.get('ratios'),
        'split_seed':   split.get('seed'),
        'metrics': {
            'reg_in_mean':   float(reg_in_arr.mean()),
            'reg_in_std':    float(reg_in_arr.std()),
            'reg_out_mean':  float(reg_out_arr.mean()),
            'reg_out_std':   float(reg_out_arr.std()),
            'delta_mean':    float(delta_arr.mean()),
            'delta_std':     float(delta_arr.std()),
            'pct_improved':  float(improved / len(results)),
            'mse_mean':      float(mse_arr.mean()),
            'mse_std':       float(mse_arr.std()),
        },
        'per_file': [{k: v for k, v in r.items() if k != 'audio_out'}
                     for r in results],
    }

    json_path = os.path.join(args.output_dir, 'test_results_v2.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nJSON saved : {json_path}')

    # ── summary plot ──────────────────────────────────────────────────────
    plot_path = os.path.join(args.output_dir, 'test_summary_v2.png')
    plot_summary(results, plot_path)
    print(f'Plot saved : {plot_path}')


if __name__ == '__main__':
    main()
