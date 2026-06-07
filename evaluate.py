# evaluate.py — Full evaluation with Rate-Distortion curves
# Generates R-D plots comparable to OpenDVC performance figures.


import os, sys, argparse, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.dvc_model import DVCModel
from utils.metrics import psnr, ssim, residual_entropy
from utils.visualization import tensor_to_uint8, flow_to_color, plot_training_history, plot_rd_curve
from models.flow_net import bilinear_warp
from data.dataset import FramePairDataset

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', type=str, nargs='+', required=True)
    p.add_argument('--labels',      type=str, nargs='+', default=None,
                   help='Labels for legend (e.g. lambda values)')
    p.add_argument('--data_root',   type=str, default='data/real_test')
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

                vis_samples.append({
                    'ref':      tensor_to_uint8(ref[i]),
                    'cur':      tensor_to_uint8(cur[i]),
                    'pred':     tensor_to_uint8(pred[0]),
                    'rec':      tensor_to_uint8(frame_rec[i]),
                    'flow':     flow_to_color(flow),
                    'residual': tensor_to_uint8(residual[0].abs().clamp(0, 1)),
                    'psnr': p, 'ssim': s, 'bpp': b, 'entropy': ent,
                })

    return {
        'psnr':    {'mean': np.mean(all_psnr), 'std': np.std(all_psnr)},
        'ssim':    {'mean': np.mean(all_ssim), 'std': np.std(all_ssim)},
        'bpp':     {'mean': np.mean(all_bpp),  'std': np.std(all_bpp)},
        'n': len(all_psnr),
    }, vis_samples

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

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
             if args.device == 'auto' else torch.device(args.device)

    if not os.path.exists(os.path.join(args.data_root, 'ref')):
        raise SystemExit(
            f"No frames at {args.data_root}/ref. Run: python extract_custom_videos.py")
    dataset = FramePairDataset(args.data_root, args.height, args.width)
    # batch_size=1 so the per-frame bpp from the model matches the per-frame
    # PSNR/SSIM (with a batch, losses['bpp'] is averaged and cannot be assigned
    # back to individual samples).
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

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

    # Reuse the canonical R-D plotter from utils.visualization (one point per
    # checkpoint/lambda). points_list: one curve per model, each a list of
    # flat {bpp, psnr, ssim} dicts.
    rd_points = [[{'bpp': r['bpp']['mean'], 'psnr': r['psnr']['mean'],
                   'ssim': r['ssim']['mean']}] for r in all_results]
    plot_rd_curve(rd_points, labels, os.path.join(args.output_dir, 'rd_curve.png'))
    print(f"R-D curve saved: {args.output_dir}/rd_curve.png")

    if all_vis:
        plot_visualization_grid(all_vis[:args.num_vis],
                                os.path.join(args.output_dir, 'sample_grid.png'))

    hist_path = os.path.join(os.path.dirname(args.checkpoints[0]), 'history.json')
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        plot_training_history(history,
                              os.path.join(args.output_dir, 'training_curves.png'))
        print(f"Training curves saved: {args.output_dir}/training_curves.png")

    print(f"\nAll results saved to: {args.output_dir}/")

if __name__ == '__main__':
    main()