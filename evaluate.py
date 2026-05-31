# evaluate.py — Full evaluation with Rate-Distortion curves
# Generates R-D plots comparable to OpenDVC performance figures.


import os, sys, argparse, json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.dvc_model import DVCModel
from utils.metrics import psnr, ssim, evaluate_frame, bd_rate, residual_entropy
from utils.visualization import tensor_to_uint8, flow_to_color, residual_heatmap, plot_training_history, plot_rd_curve
from models.flow_net import bilinear_warp
from data.dataset import FramePairDataset, generate_synthetic_frames

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', type=str, nargs='+', required=True)
    p.add_argument('--labels',      type=str, nargs='+', default=None,
                   help='Labels for legend (e.g. lambda values)')
    p.add_argument('--data_root',   type=str, default='data/synthetic')
    p.add_argument('--height',      type=int, default=128)
    p.add_argument('--width',       type=int, default=128)
    p.add_argument('--output_dir',  type=str, default='results')
    p.add_argument('--num_vis',     type=int, default=6)
    p.add_argument('--device',      type=str, default='auto')
    return p.parse_args()

def load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get('args', {})
    model = DVCModel(
        flow_M=cfg.get('flow_M', 128),
        res_M=cfg.get('res_M',  128),
        lmbda=cfg.get('lmbda',  512),
    )
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device).eval()
    return model, cfg

@torch.no_grad()
def evaluate_model(model, loader, device, num_vis=6):
    """Run evaluation, collect metrics and visualization samples."""
    all_psnr, all_ssim, all_bpp, all_entropy = [], [], [], []
    vis_samples = []

    for ref, cur in tqdm(loader, desc='  eval', leave=False):
        ref, cur = ref.to(device), cur.to(device)
        frame_rec, losses = model(ref, cur)

        for i in range(ref.size(0)):
            p  = psnr(cur[i:i+1], frame_rec[i:i+1])
            s  = ssim(cur[i:i+1], frame_rec[i:i+1])
            b  = losses['bpp'].item()
            all_psnr.append(p)
            all_ssim.append(s)
            all_bpp.append(b)

            if len(vis_samples) < num_vis:

                flow = model.flow_net(ref[i:i+1], cur[i:i+1])
                pred = bilinear_warp(ref[i:i+1], flow)
                residual = cur[i:i+1] - pred
                ent = residual_entropy(residual[0].cpu())

                def t2np(t):
                    if t.dim() == 4: t = t[0]
                    return (t.cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)

                flow_np = flow[0].cpu().permute(1,2,0).numpy()
                u, v = flow_np[:,:,0], flow_np[:,:,1]
                mag = np.sqrt(u**2 + v**2)
                ang = np.arctan2(v, u)
                hue = ((ang + np.pi) / (2*np.pi) * 179).astype(np.uint8)
                sat = np.ones_like(hue) * 255
                val_c = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)
                flow_color = cv2.cvtColor(np.stack([hue, sat, val_c], 2), cv2.COLOR_HSV2RGB)

                vis_samples.append({
                    'ref':      t2np(ref[i]),
                    'cur':      t2np(cur[i]),
                    'pred':     t2np(pred[0]),
                    'rec':      t2np(frame_rec[i]),
                    'flow':     flow_color,
                    'residual': t2np(residual[0].abs().clamp(0,1)),
                    'psnr': p, 'ssim': s, 'bpp': b, 'entropy': ent,
                })

    return {
        'psnr':    {'mean': np.mean(all_psnr), 'std': np.std(all_psnr)},
        'ssim':    {'mean': np.mean(all_ssim), 'std': np.std(all_ssim)},
        'bpp':     {'mean': np.mean(all_bpp),  'std': np.std(all_bpp)},
        'n': len(all_psnr),
    }, vis_samples

def plot_rd_curves(rd_points, labels, save_path):
    """Plot Rate-Distortion curves (like OpenDVC performance figures)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2563EB', '#DC2626', '#059669', '#D97706', '#7C3AED']

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx, (points, label) in enumerate(zip(rd_points, labels)):
        bpp_vals  = [p['bpp']['mean'] for p in points]
        psnr_vals = [p['psnr']['mean'] for p in points]
        ssim_vals = [p['ssim']['mean'] for p in points]
        c = colors[idx % len(colors)]

        axes[0].plot(bpp_vals, psnr_vals, 'o-', color=c, label=label, linewidth=2, markersize=6)
        axes[1].plot(bpp_vals, ssim_vals, 'o-', color=c, label=label, linewidth=2, markersize=6)

    axes[0].set_xlabel('Bits per pixel (bpp)', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('Rate-Distortion (PSNR)', fontsize=13)
    axes[0].legend(title='λ', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Bits per pixel (bpp)', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('Rate-Distortion (SSIM)', fontsize=13)
    axes[1].legend(title='λ', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Learned Motion Compensation — R-D Performance', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"R-D curve saved: {save_path}")

def plot_visualization_grid(vis_samples, save_path):
    """Sample grid: ref | cur | pred | rec | flow | residual."""
    n = len(vis_samples)
    fig, axes = plt.subplots(n, 6, figsize=(22, 3.5 * n))
    if n == 1: axes = [axes]

    titles = ['Reference $I_{t-1}$', 'Current $I_t$', 'Predicted $\\hat{I}_t$\n(warp only)',
              'Reconstructed $I_t$\n(+residual)', 'Optical Flow', 'Residual $|R_t|$']
    for j, t in enumerate(titles):
        axes[0][j].set_title(t, fontsize=9, pad=5)

    for i, d in enumerate(vis_samples):
        imgs = [d['ref'], d['cur'], d['pred'], d['rec'], d['flow'], d['residual']]
        for j, img in enumerate(imgs):
            axes[i][j].imshow(img)
            axes[i][j].axis('off')
        axes[i][0].set_ylabel(
            f"PSNR:{d['psnr']:.1f}dB\nSSIM:{d['ssim']:.3f}\nbpp:{d['bpp']:.4f}",
            fontsize=8, rotation=0, labelpad=70, va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Visualization grid saved: {save_path}")

def plot_training_curves(history_path, save_path):
    """Plot training history."""
    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    keys = [('loss', 'Total Loss'), ('psnr', 'PSNR (dB)'), ('bpp', 'bpp')]

    for ax, (key, ylabel) in zip(axes, keys):
        t_vals = [e.get(key, 0) for e in history.get('train', [])]
        v_vals = [e.get(key, 0) for e in history.get('val', [])]
        epochs = list(range(1, len(t_vals) + 1))
        ax.plot(epochs, t_vals, label='Train', color='#2563EB')
        ax.plot(epochs, v_vals, label='Val',   color='#DC2626')
        ax.set_title(ylabel, fontsize=11)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Training Curves — DVC-inspired Learned Motion Compensation', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {save_path}")

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
             if args.device == 'auto' else torch.device(args.device)

    if not os.path.exists(os.path.join(args.data_root, 'ref')):
        generate_synthetic_frames(300, args.height, args.width, args.data_root)
    dataset = FramePairDataset(args.data_root, args.height, args.width)
    loader  = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    labels = args.labels or [os.path.basename(os.path.dirname(c)) for c in args.checkpoints]

    all_results = []
    all_vis     = []

    print(f"Evaluating {len(args.checkpoints)} checkpoint(s) on {len(dataset)} pairs...")

    for ckpt_path, label in zip(args.checkpoints, labels):
        print(f"\n[{label}] {ckpt_path}")
        model, cfg = load_model_from_ckpt(ckpt_path, device)
        results, vis = evaluate_model(model, loader, device, args.num_vis)
        all_results.append(results)
        all_vis.extend(vis[:min(2, len(vis))])  # 2 samples per checkpoint

        print(f"  PSNR: {results['psnr']['mean']:.2f} ± {results['psnr']['std']:.2f} dB")
        print(f"  SSIM: {results['ssim']['mean']:.4f} ± {results['ssim']['std']:.4f}")
        print(f"  bpp:  {results['bpp']['mean']:.5f} ± {results['bpp']['std']:.5f}")

    summary = {lbl: res for lbl, res in zip(labels, all_results)}
    with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    plot_rd_curves([[r] for r in all_results], labels,
                   os.path.join(args.output_dir, 'rd_curve.png'))

    if all_vis:
        plot_visualization_grid(all_vis[:args.num_vis],
                                os.path.join(args.output_dir, 'sample_grid.png'))

    hist_path = os.path.join(os.path.dirname(args.checkpoints[0]), 'history.json')
    if os.path.exists(hist_path):
        plot_training_curves(hist_path,
                             os.path.join(args.output_dir, 'training_curves.png'))

    print(f"\nAll results saved to: {args.output_dir}/")

if __name__ == '__main__':
    main()