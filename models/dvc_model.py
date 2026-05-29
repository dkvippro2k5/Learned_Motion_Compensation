"""
models/dvc_model.py
Full DVC-inspired Learned Video Compression model.

Combines:
  - PWCNet for optical flow estimation
  - MotionCompressor for encoding/decoding flow
  - BilinearWarp for P-frame prediction
  - ResidualCompressor for encoding/decoding residuals
  - IFrameCodec (CompressAI) for I-frames

Rate-Distortion training objective (from DVC paper):
    L = λ · D + R_motion + R_residual
where:
    D  = distortion (MSE or MS-SSIM between I_t and reconstructed I_t)
    R  = estimated bit-rate from entropy bottleneck
    λ  = Lagrangian multiplier (controls rate-distortion tradeoff)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .flow_net import PWCNet, bilinear_warp
from .entropy_coder import MotionCompressor, ResidualCompressor, IFrameCodec, rate_estimate


class DVCModel(nn.Module):
    """
    End-to-end learned video compression model (P-frame codec).

    Training mode: forward() returns reconstructed frame + losses
    Inference mode: encode() / decode() write/read actual bitstreams

    Args:
        flow_M: latent channels for motion compressor (default 128)
        res_M:  latent channels for residual compressor (default 128)
        lmbda:  rate-distortion lambda (higher = more quality, more bits)
        use_iframe_codec: if True, also encode I-frames with learned codec
    """
    def __init__(self, flow_M=128, res_M=128, lmbda=512, use_iframe_codec=False):
        super().__init__()
        self.lmbda = lmbda

        # Flow estimation (not compressed, guides the motion coder)
        self.flow_net = PWCNet(max_disp=4)

        # Motion compression
        self.motion_coder = MotionCompressor(M=flow_M)

        # Residual compression
        self.residual_coder = ResidualCompressor(M=res_M)

        # Optional I-frame codec (CompressAI)
        self.use_iframe_codec = use_iframe_codec
        if use_iframe_codec:
            self.iframe_codec = IFrameCodec(quality=4, pretrained=True)

    def forward(self, frame_ref, frame_cur):
        """
        Training forward pass for one P-frame.

        Args:
            frame_ref: previous (reference) frame (B, 3, H, W) in [0, 1]
            frame_cur: current frame to compress  (B, 3, H, W) in [0, 1]
        Returns:
            frame_rec: reconstructed current frame (B, 3, H, W)
            losses:    dict with rate/distortion breakdown
        """
        B, C, H, W = frame_cur.shape
        num_pixels = B * H * W

        # ── Step 1: Estimate optical flow ──────────────────────────────────
        flow_raw = self.flow_net(frame_ref, frame_cur)

        # ── Step 2: Compress and reconstruct flow ──────────────────────────
        flow_hat, motion_likelihoods = self.motion_coder(flow_raw)

        # ── Step 3: Warp reference frame using compressed flow ──────────────
        frame_pred = bilinear_warp(frame_ref, flow_hat)

        # ── Step 4: Compute residual ───────────────────────────────────────
        residual = frame_cur - frame_pred

        # ── Step 5: Compress and reconstruct residual ──────────────────────
        residual_hat, res_likelihoods = self.residual_coder(residual)

        # ── Step 6: Reconstruct final frame ───────────────────────────────
        frame_rec = (frame_pred + residual_hat).clamp(0, 1)

        # ── Rate-Distortion loss ───────────────────────────────────────────
        R_motion   = rate_estimate(motion_likelihoods, num_pixels)
        R_residual = rate_estimate(res_likelihoods,    num_pixels)
        R_total    = R_motion + R_residual

        # Distortion: MSE
        D_mse = F.mse_loss(frame_rec, frame_cur)

        # Distortion: MS-SSIM (1 - ms_ssim, so lower = better)
        D_msssim = 1.0 - ms_ssim(frame_rec, frame_cur)

        # Total loss: λ·D + R  (use MSE by default; switch to MS-SSIM by passing metric)
        loss_psnr   = self.lmbda * D_mse   + R_total
        loss_msssim = self.lmbda * D_msssim + R_total

        return frame_rec, {
            'loss_psnr':   loss_psnr,
            'loss_msssim': loss_msssim,
            'R_motion':    R_motion,
            'R_residual':  R_residual,
            'R_total':     R_total,
            'D_mse':       D_mse,
            'D_msssim':    D_msssim,
            'bpp':         R_total,
        }

    @torch.no_grad()
    def encode_frame(self, frame_ref, frame_cur):
        """
        Compress one P-frame to actual byte strings (inference).

        Returns:
            bitstreams: dict with 'motion' and 'residual' byte strings
            meta:       dict with shape info for decoder
            frame_rec:  reconstructed frame (for next frame reference)
        """
        B, C, H, W = frame_cur.shape

        # Estimate flow
        flow_raw = self.flow_net(frame_ref, frame_cur)

        # Compress motion
        motion_strings, motion_shape = self.motion_coder.compress(flow_raw)

        # Reconstruct flow for warping
        flow_hat = self.motion_coder.decompress(motion_strings, motion_shape, (H, W))

        # Warp and compute residual
        frame_pred = bilinear_warp(frame_ref, flow_hat)
        residual = frame_cur - frame_pred

        # Compress residual
        res_strings, res_shape = self.residual_coder.compress(residual)

        # Reconstruct
        residual_hat = self.residual_coder.decompress(res_strings, res_shape, (H, W))
        frame_rec = (frame_pred + residual_hat).clamp(0, 1)

        return {
            'motion_strings':  motion_strings,
            'residual_strings': res_strings,
            'motion_shape':    motion_shape,
            'res_shape':       res_shape,
            'H': H, 'W': W,
        }, frame_rec

    @torch.no_grad()
    def decode_frame(self, frame_ref, bitstream_dict):
        """
        Decode a P-frame from byte strings.

        Args:
            frame_ref:       previous decoded reference frame
            bitstream_dict:  output of encode_frame()
        Returns:
            frame_rec: decoded current frame
        """
        H = bitstream_dict['H']
        W = bitstream_dict['W']

        # Decompress flow
        flow_hat = self.motion_coder.decompress(
            bitstream_dict['motion_strings'],
            bitstream_dict['motion_shape'],
            (H, W)
        )

        # Warp
        frame_pred = bilinear_warp(frame_ref, flow_hat)

        # Decompress residual
        residual_hat = self.residual_coder.decompress(
            bitstream_dict['residual_strings'],
            bitstream_dict['res_shape'],
            (H, W)
        )

        return (frame_pred + residual_hat).clamp(0, 1)


# ── MS-SSIM implementation (no external dependency) ──────────────────────────

def _gaussian_window(size=11, sigma=1.5, channels=3):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    return kernel.expand(channels, 1, size, size)


def ms_ssim(img1, img2, levels=3, window_size=11):
    """
    Multi-Scale Structural Similarity (MS-SSIM).
    Returns value in [0, 1] — higher is better.
    """
    weights = torch.tensor([0.0448, 0.2856, 0.3001], device=img1.device)
    weights = weights / weights.sum()

    mssim = []
    C, H, W = img1.shape[1], img1.shape[2], img1.shape[3]

    for i in range(levels):
        win = _gaussian_window(window_size, 1.5, C).to(img1.device)
        pad = window_size // 2

        mu1 = F.conv2d(img1, win, groups=C, padding=pad)
        mu2 = F.conv2d(img2, win, groups=C, padding=pad)
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2

        sig1 = F.conv2d(img1**2, win, groups=C, padding=pad) - mu1_sq
        sig2 = F.conv2d(img2**2, win, groups=C, padding=pad) - mu2_sq
        sig12 = F.conv2d(img1*img2, win, groups=C, padding=pad) - mu1_mu2

        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sig12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1)*(sig1 + sig2 + C2))
        mssim.append(ssim_map.mean())

        # Downsample for next level
        if i < levels - 1:
            img1 = F.avg_pool2d(img1, 2)
            img2 = F.avg_pool2d(img2, 2)

    val = torch.stack(mssim)
    return (val * weights).sum()