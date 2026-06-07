# Evaluation metrics for video compression
# PSNR, MS-SSIM, bits-per-pixel (bpp), BD-Rate


import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_sk
import math

def psnr(img1, img2, max_val=1.0):
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

def ms_ssim_numpy(img1_np, img2_np, levels=3):
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

def bitstream_bpp(byte_strings, H: int, W: int) -> float:
    """Real bits-per-pixel from an actual compressed bitstream. Recursively
    sums the lengths of all byte-strings (handles the nested lists produced by
    the hyperprior codecs, e.g. [[y_strings], [z_strings]])."""
    def _count(x):
        if isinstance(x, (bytes, bytearray)):
            return len(x)
        if isinstance(x, (list, tuple)):
            return sum(_count(e) for e in x)
        return 0
    return _count(byte_strings) * 8 / (H * W)

def bd_rate(rate1, psnr1, rate2, psnr2):
    def _interp(rates, psnrs):
        log_rates = np.log(np.array(rates, dtype=np.float64))
        psnrs_arr = np.array(psnrs, dtype=np.float64)

        # Standard BD uses a cubic fit (needs >=4 points); fall back to a
        # lower-degree fit when fewer operating points are available.
        deg = min(3, len(psnrs_arr) - 1)
        coeffs = np.polyfit(psnrs_arr, log_rates, deg)
        return coeffs, (psnrs_arr.min(), psnrs_arr.max())

    coeffs1, (lo1, hi1) = _interp(rate1, psnr1)
    coeffs2, (lo2, hi2) = _interp(rate2, psnr2)

    lo = max(lo1, lo2)
    hi = min(hi1, hi2)
    if lo >= hi:
        return float('nan')

    psnr_pts = np.linspace(lo, hi, 100)
    rate_diff = np.polyval(coeffs2, psnr_pts) - np.polyval(coeffs1, psnr_pts)
    avg_log_rate_diff = np.trapz(rate_diff, psnr_pts) / (hi - lo)

    return float((np.exp(avg_log_rate_diff) - 1) * 100)

def residual_entropy(residual, bins=256):
    if isinstance(residual, torch.Tensor):
        residual = residual.detach().cpu().numpy()
    r = np.clip((residual + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(r.flatten(), bins=bins, range=(0, 256))
    hist = hist[hist > 0].astype(np.float64)
    prob = hist / hist.sum()
    return float(-np.sum(prob * np.log2(prob)))

def evaluate_frame(frame_cur, frame_rec, bpp_val=None, residual=None):
 
    results = {
        'psnr': psnr(frame_cur, frame_rec),
        'ssim': ssim(frame_cur, frame_rec),
    }

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