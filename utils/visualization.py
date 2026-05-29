"""
utils/visualization.py — Visualization helpers for motion compensation outputs.
Converts tensors → numpy images ready for matplotlib / cv2.
"""

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(B,C,H,W) or (C,H,W) float [0,1] tensor → (H,W,3) uint8 numpy."""
    if t.dim() == 4:
        t = t[0]
    img = t.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def flow_to_color(flow: torch.Tensor) -> np.ndarray:
    """
    Convert (2,H,W) or (1,2,H,W) flow tensor to HSV-colored RGB image.
    Hue = direction, Saturation = 255, Value = magnitude (normalized).
    """
    if flow.dim() == 4:
        flow = flow[0]
    flow_np = flow.detach().cpu().numpy()
    u, v = flow_np[0], flow_np[1]

    mag = np.sqrt(u ** 2 + v ** 2)
    ang = np.arctan2(v, u)  # [-pi, pi]

    hue = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    sat = np.ones_like(hue, dtype=np.uint8) * 255
    val = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)

    hsv = np.stack([hue, sat, val], axis=2)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def residual_heatmap(residual: torch.Tensor) -> np.ndarray:
    """
    Convert residual (C,H,W) to a heatmap.
    Bright = large error, dark = small error.
    """
    if residual.dim() == 4:
        residual = residual[0]
    r = residual.detach().cpu().abs().mean(dim=0).numpy()  # (H,W)
    r_norm = (r / (r.max() + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(r_norm, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def make_comparison_grid(ref, cur, pred, frame_rec, flow, residual,
                          metrics: dict, save_path: str):
    """
    Save a 2-row comparison figure:
    Row 1: Reference | Current | Predicted (warp only) | Reconstructed
    Row 2: Optical Flow | Residual heatmap | Metrics text | Entropy hist
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    ref_np  = tensor_to_uint8(ref)
    cur_np  = tensor_to_uint8(cur)
    pred_np = tensor_to_uint8(pred)
    rec_np  = tensor_to_uint8(frame_rec)
    flow_np = flow_to_color(flow)
    heat_np = residual_heatmap(residual)

    fig = plt.figure(figsize=(22, 10), facecolor='#0f1117')
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.06)

    # Row 1
    panels_r1 = [
        (ref_np,  'Reference $I_{t-1}$',         '#60A5FA'),
        (cur_np,  'Current $I_t$',                '#34D399'),
        (pred_np, 'Predicted $\\hat{I}_t$ (warp)', '#A78BFA'),
        (rec_np,  'Reconstructed $I_t$ (+resid)', '#FB923C'),
    ]
    for col, (img, title, color) in enumerate(panels_r1):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_title(title, color=color, fontsize=11, pad=5)
        ax.axis('off')

    # Row 2 col 0: Flow
    ax_flow = fig.add_subplot(gs[1, 0])
    ax_flow.imshow(flow_np)
    ax_flow.set_title('Optical Flow', color='#FB923C', fontsize=11, pad=5)
    ax_flow.axis('off')

    # Row 2 col 1: Heatmap
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_heat.imshow(heat_np)
    ax_heat.set_title('Residual Heatmap', color='#F87171', fontsize=11, pad=5)
    ax_heat.axis('off')

    # Row 2 col 2: Metrics
    ax_met = fig.add_subplot(gs[1, 2])
    ax_met.set_facecolor('#1e2130')
    ax_met.axis('off')
    ax_met.text(0.5, 0.92, 'Metrics', color='white', fontsize=13,
                fontweight='bold', ha='center', va='top', transform=ax_met.transAxes)
    items = [
        ('PSNR',     f"{metrics.get('psnr', 0):.2f} dB",  '#60A5FA'),
        ('SSIM',     f"{metrics.get('ssim', 0):.4f}",      '#34D399'),
        ('bpp',      f"{metrics.get('bpp', 0):.5f}",       '#FB923C'),
        ('R_motion', f"{metrics.get('R_motion', 0):.5f}",  '#A78BFA'),
        ('R_resid',  f"{metrics.get('R_residual', 0):.5f}",'#F87171'),
    ]
    for i, (name, val, c) in enumerate(items):
        y = 0.72 - i * 0.15
        ax_met.text(0.1, y,       name, color='#9CA3AF', fontsize=10,
                    transform=ax_met.transAxes)
        ax_met.text(0.1, y - 0.07, val, color=c, fontsize=13,
                    fontweight='bold', transform=ax_met.transAxes)

    # Row 2 col 3: Residual histogram
    ax_hist = fig.add_subplot(gs[1, 3])
    ax_hist.set_facecolor('#1e2130')
    if residual.dim() == 4:
        res_data = residual[0]
    else:
        res_data = residual
    r_flat = res_data.detach().cpu().numpy().flatten()
    ax_hist.hist(r_flat, bins=60, color='#F87171', alpha=0.8, edgecolor='none')
    ax_hist.set_title('Residual Distribution', color='white', fontsize=10, pad=4)
    ax_hist.set_xlabel('Value', color='#9CA3AF', fontsize=9)
    ax_hist.set_ylabel('Count',  color='#9CA3AF', fontsize=9)
    ax_hist.tick_params(colors='#9CA3AF', labelsize=8)
    for sp in ['top', 'right']:
        ax_hist.spines[sp].set_visible(False)
    ax_hist.spines['bottom'].set_color('#374151')
    ax_hist.spines['left'].set_color('#374151')

    plt.suptitle('Learned Motion Compensation — Pipeline Output',
                 color='white', fontsize=14, y=1.01)
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#0f1117')
    plt.close()


def plot_training_history(history: dict, save_path: str):
    """Plot training & validation curves for loss, PSNR, bpp."""
    metrics_to_plot = [
        ('loss',  'Total Loss',     '#2563EB'),
        ('psnr',  'PSNR (dB)',      '#059669'),
        ('bpp',   'bpp',            '#D97706'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, ylabel, color) in zip(axes, metrics_to_plot):
        t_vals = [e.get(key, 0) for e in history.get('train', [])]
        v_vals = [e.get(key, 0) for e in history.get('val',   [])]
        eps    = list(range(1, len(t_vals) + 1))
        ax.plot(eps, t_vals, label='Train', color=color,        linewidth=2)
        ax.plot(eps, v_vals, label='Val',   color=color, alpha=0.5,
                linewidth=2, linestyle='--')
        ax.set_title(ylabel, fontsize=11)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.suptitle('Training History — DVC-inspired Learned Motion Compensation',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_rd_curve(points_list: list, labels: list, save_path: str):
    """
    Plot Rate-Distortion curves (PSNR & SSIM vs bpp).
    points_list: list of lists, each inner list = one model's R-D points.
    Each point is a dict {'bpp': ..., 'psnr': ..., 'ssim': ...}.
    """
    colors = ['#2563EB', '#DC2626', '#059669', '#D97706', '#7C3AED', '#DB2777']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2)

    for idx, (pts, lbl) in enumerate(zip(points_list, labels)):
        bpps  = [p['bpp']  for p in pts]
        psnrs = [p['psnr'] for p in pts]
        ssims = [p['ssim'] for p in pts]
        c = colors[idx % len(colors)]
        # Sort by bpp for clean curve
        order = sorted(range(len(bpps)), key=lambda i: bpps[i])
        bpps  = [bpps[i]  for i in order]
        psnrs = [psnrs[i] for i in order]
        ssims = [ssims[i] for i in order]

        axes[0].plot(bpps, psnrs, 'o-', color=c, label=lbl, lw=2, ms=6)
        axes[1].plot(bpps, ssims, 'o-', color=c, label=lbl, lw=2, ms=6)

    axes[0].set_xlabel('Bits per pixel (bpp)', fontsize=11)
    axes[0].set_ylabel('PSNR (dB)', fontsize=11)
    axes[0].set_title('R-D Curve — PSNR', fontsize=12)
    axes[0].legend(title='Model / λ', fontsize=9)

    axes[1].set_xlabel('Bits per pixel (bpp)', fontsize=11)
    axes[1].set_ylabel('SSIM', fontsize=11)
    axes[1].set_title('R-D Curve — SSIM', fontsize=12)
    axes[1].legend(title='Model / λ', fontsize=9)

    plt.suptitle('Rate-Distortion Performance', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()