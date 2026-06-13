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

