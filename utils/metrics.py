"""
utils/metrics.py — Evaluation metrics for video compression
PSNR, MS-SSIM, bits-per-pixel (bpp), BD-Rate
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_sk
import math


# ── PSNR ─────────────────────────────────────────────────────────────────────

def psnr(img1, img2, max_val=1.0):
    """PSNR in dB. Higher = better. > 30 dB good, > 40 dB excellent."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu()
        img2 = img2.detach().cpu()
        if img1.dim() == 4:
            img1, img2 = img1[0], img2[0]
        img1 = img1.permute(1, 2, 0).numpy()
        img2 = img2.permute(1, 2, 0).numpy()
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return float(20 * np.log10(max_val / math.sqrt(mse)))


# ── MS-SSIM ──────────────────────────────────────────────────────────────────

def ms_ssim_numpy(img1_np, img2_np, levels=3):
    """
    MS-SSIM computed on numpy arrays (H, W, C) float32 in [0, 1].
    Returns scalar in [0, 1].
    """
    weights = np.array([0.0448, 0.2856, 0.3001])
    weights /= weights.sum()

    mssim = []
    img1, img2 = img1_np.copy(), img2_np.copy()

    for i in range(levels):
        s = ssim_sk(img1, img2, data_range=1.0, channel_axis=-1)
        mssim.append(s)
        if i < levels - 1:
            from skimage.transform import resize
            h, w = img1.shape[:2]
            img1 = resize(img1, (h // 2, w // 2), anti_aliasing=True)
            img2 = resize(img2, (h // 2, w // 2), anti_aliasing=True)

    return float(np.dot(np.array(mssim), weights))


def ssim(img1, img2, max_val=1.0):
    """Single-scale SSIM (kept for compatibility)."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu()
        if img1.dim() == 4: img1 = img1[0]
        img1 = img1.permute(1, 2, 0).numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu()
        if img2.dim() == 4: img2 = img2[0]
        img2 = img2.permute(1, 2, 0).numpy()
    return float(ssim_sk(img1, img2, data_range=max_val, channel_axis=-1))


# ── Bits per pixel ────────────────────────────────────────────────────────────

def compute_bpp(likelihoods_tensor, num_pixels: int) -> float:
    """
    Compute actual bits-per-pixel from likelihood tensor.

    Args:
        likelihoods_tensor: probability tensor from entropy model
        num_pixels: H * W (NOT batch * H * W, per-image bpp)
    Returns:
        bpp: float
    """
    bits = -torch.log2(likelihoods_tensor.clamp(min=1e-9)).sum().item()
    return bits / num_pixels


def bitstream_bpp(byte_strings, H: int, W: int) -> float:
    """
    Compute actual bpp from compressed byte strings (inference).
    More accurate than estimate — uses real entropy coding output.
    """
    if isinstance(byte_strings, (list, tuple)):
        total_bytes = sum(len(s) for s in byte_strings
                          if isinstance(s, (bytes, bytearray)))
    else:
        total_bytes = len(byte_strings)
    return total_bytes * 8 / (H * W)


# ── BD-Rate ───────────────────────────────────────────────────────────────────

def bd_rate(rate1, psnr1, rate2, psnr2):
    """
    Compute Bjøntegaard Delta Rate (BD-Rate) between two R-D curves.
    A negative BD-Rate means method 2 uses fewer bits for the same quality.

    Args:
        rate1: list of bpp values for method 1 (at least 4 points)
        psnr1: list of PSNR values for method 1
        rate2: list of bpp values for method 2
        psnr2: list of PSNR values for method 2
    Returns:
        bd_rate_percent: float — positive = method 2 is worse
    """
    import numpy.polynomial.polynomial as P

    def _interp(rates, psnrs):
        log_rates = np.log(np.array(rates, dtype=np.float64))
        psnrs_arr = np.array(psnrs, dtype=np.float64)
        # Fit cubic polynomial: rate as function of PSNR
        coeffs = np.polyfit(psnrs_arr, log_rates, 3)
        return coeffs, (psnrs_arr.min(), psnrs_arr.max())

    coeffs1, (lo1, hi1) = _interp(rate1, psnr1)
    coeffs2, (lo2, hi2) = _interp(rate2, psnr2)

    # Integration range: intersection of both PSNR ranges
    lo = max(lo1, lo2)
    hi = min(hi1, hi2)
    if lo >= hi:
        return float('nan')

    # Integrate rate difference over PSNR range
    psnr_pts = np.linspace(lo, hi, 100)
    rate_diff = np.polyval(coeffs2, psnr_pts) - np.polyval(coeffs1, psnr_pts)
    avg_log_rate_diff = np.trapz(rate_diff, psnr_pts) / (hi - lo)

    return float((np.exp(avg_log_rate_diff) - 1) * 100)


# ── Residual quality ──────────────────────────────────────────────────────────

def residual_entropy(residual, bins=256):
    """Shannon entropy of residual (bits). Lower = more compressible."""
    if isinstance(residual, torch.Tensor):
        residual = residual.detach().cpu().numpy()
    r = np.clip((residual + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(r.flatten(), bins=bins, range=(0, 256))
    hist = hist[hist > 0].astype(np.float64)
    prob = hist / hist.sum()
    return float(-np.sum(prob * np.log2(prob)))


# ── Unified evaluation ────────────────────────────────────────────────────────

def evaluate_frame(frame_cur, frame_rec, bpp_val=None, residual=None):
    """
    Full evaluation of a reconstructed P-frame.

    Returns:
        dict with: psnr, ssim, ms_ssim, (bpp if provided), (entropy if residual given)
    """
    results = {
        'psnr': psnr(frame_cur, frame_rec),
        'ssim': ssim(frame_cur, frame_rec),
    }

    # MS-SSIM (numpy)
    if isinstance(frame_cur, torch.Tensor):
        i1 = frame_cur.detach().cpu()
        i2 = frame_rec.detach().cpu()
        if i1.dim() == 4: i1, i2 = i1[0], i2[0]
        i1 = i1.permute(1, 2, 0).numpy()
        i2 = i2.permute(1, 2, 0).numpy()
    else:
        i1, i2 = frame_cur, frame_rec
    results['ms_ssim'] = ms_ssim_numpy(i1, i2)

    if bpp_val is not None:
        results['bpp'] = bpp_val
    if residual is not None:
        results['residual_entropy'] = residual_entropy(residual)

    return results