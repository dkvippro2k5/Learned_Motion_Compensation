"""
warp.py — Warping & Residual computation
Implements bilinear warping and residual calculation for motion compensation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


def bilinear_warp(frame, flow):
    """
    Warp a frame using a dense optical flow field (bilinear interpolation).
    
    Args:
        frame: Reference frame tensor (B, C, H, W), values in [0, 1]
        flow:  Optical flow field    (B, 2, H, W)
               flow[:,0] = horizontal displacement (u)
               flow[:,1] = vertical displacement   (v)
    Returns:
        warped: Warped frame (B, C, H, W)
    """
    B, C, H, W = frame.shape

    # Build base grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=frame.device),
        torch.arange(W, dtype=torch.float32, device=frame.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

    # Apply flow displacement
    new_grid = grid + flow

    # Normalize to [-1, 1] for grid_sample
    new_grid[:, 0] = 2.0 * new_grid[:, 0] / max(W - 1, 1) - 1.0
    new_grid[:, 1] = 2.0 * new_grid[:, 1] / max(H - 1, 1) - 1.0

    new_grid = new_grid.permute(0, 2, 3, 1)  # (B, H, W, 2)

    warped = F.grid_sample(
        frame, new_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    return warped


def compute_residual(frame_cur, frame_pred):
    """
    Compute residual map: R_t = I_t - I_hat_t
    
    Args:
        frame_cur:  Original current frame     (B, C, H, W) or (C, H, W)
        frame_pred: Predicted (warped) frame   same shape
    Returns:
        residual: Residual tensor, same shape as input
    """
    return frame_cur - frame_pred


def residual_energy(residual):
    """
    Measure residual energy (mean absolute value).
    Lower = better motion compensation.
    
    Args:
        residual: Residual tensor (B, C, H, W)
    Returns:
        energy: Scalar tensor (mean over batch)
    """
    return residual.abs().mean()


def residual_entropy(residual_np, bins=256):
    """
    Compute Shannon entropy of residual histogram.
    Lower entropy = more compressible residual = better compression.
    
    Args:
        residual_np: Residual as numpy array (H, W) or (H, W, C), float in [-1, 1]
        bins: Number of histogram bins
    Returns:
        entropy: Float scalar (bits per symbol)
    """
    # Quantize to uint8 range [0, 255]
    r = np.clip((residual_np + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(r.flatten(), bins=bins, range=(0, 256))
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]
    prob = hist / hist.sum()
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)


def flow_to_color(flow_np):
    """
    Convert optical flow (H, W, 2) numpy array to color visualization (H, W, 3).
    Uses HSV color wheel: hue = direction, value = magnitude.
    
    Args:
        flow_np: Flow array (H, W, 2) with (u, v) channels
    Returns:
        color_flow: RGB image (H, W, 3) uint8
    """
    u = flow_np[:, :, 0]
    v = flow_np[:, :, 1]

    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)  # radians in [-pi, pi]

    # Normalize
    mag_norm = np.clip(magnitude / (magnitude.max() + 1e-8), 0, 1)
    hue = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    sat = np.ones_like(hue) * 255
    val = (mag_norm * 255).astype(np.uint8)

    hsv = np.stack([hue, sat, val], axis=2)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def tensor_to_numpy(tensor):
    """Convert (B, C, H, W) or (C, H, W) tensor to (H, W, C) uint8 numpy."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img