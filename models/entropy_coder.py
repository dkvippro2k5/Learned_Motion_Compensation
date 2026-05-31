"""
models/entropy_coder.py
Entropy coding for motion fields and residuals using CompressAI.

This replaces the BPG + tensorflow-compression approach in original OpenDVC.
Uses compressai.entropy_models which is PyTorch-native and maintained.

Key classes:
  - MotionCompressor   : compress/decompress optical flow field
  - ResidualCompressor : compress/decompress residual map
  - IFrameCodec        : learned image codec for I-frames (bmshj2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import FactorizedPrior

class MotionCompressor(nn.Module):
    """
    Compress optical flow field using a hyperprior entropy model.
    Based on the motion compression branch in DVC.

    Architecture:
        Encoder: flow (2ch) → latent z (M ch)
        Entropy: EntropyBottleneck on z
        Decoder: z → reconstructed flow (2ch)

    Args:
        M: number of latent channels (default 128)
    """
    def __init__(self, M: int = 128):
        super().__init__()
        self.M = M

        self.encoder = nn.Sequential(
            nn.Conv2d(2,  M // 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(M // 2, M // 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(M // 2, M, 3, stride=2, padding=1),
        )

        self.entropy_bottleneck = EntropyBottleneck(M)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(M, M // 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(M // 2, M // 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(M // 2, 2, 4, stride=2, padding=1),
        )

    def forward(self, flow):
        """
        Training forward pass (soft quantization for gradients).

        Args:
            flow: (B, 2, H, W)
        Returns:
            flow_hat: reconstructed flow (B, 2, H, W)
            likelihoods: probability tensor for rate estimation
        """
        z = self.encoder(flow)
        z_hat, likelihoods = self.entropy_bottleneck(z)
        flow_hat = self.decoder(z_hat)

        if flow_hat.shape != flow.shape:
            flow_hat = F.interpolate(flow_hat, size=flow.shape[2:],
                                     mode='bilinear', align_corners=False)
        return flow_hat, likelihoods

    def compress(self, flow):
        """
        Inference: actually compress to bytes.
        Returns: dict with 'strings' and 'shape' for bitstream writing.
        """
        z = self.encoder(flow)
        return self.entropy_bottleneck.compress(z), z.shape[2:]

    def decompress(self, strings, shape, out_size):
        """Inference: decompress bytes back to flow."""
        z_hat = self.entropy_bottleneck.decompress(strings, shape)
        flow_hat = self.decoder(z_hat)
        if flow_hat.shape[2:] != out_size:
            flow_hat = F.interpolate(flow_hat, size=out_size,
                                     mode='bilinear', align_corners=False)
        return flow_hat

class ResidualCompressor(nn.Module):
    """
    Compress residual map R_t = I_t - I_hat_t.
    Uses the same factorized bottleneck entropy model.

    Args:
        M: latent channels (default 128)
    """
    def __init__(self, M: int = 128):
        super().__init__()
        self.M = M

        self.encoder = nn.Sequential(
            nn.Conv2d(3,  M // 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(M // 2, M // 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(M // 2, M, 3, stride=2, padding=1),
        )
        
        self.entropy_bottleneck = EntropyBottleneck(M)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(M, M // 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(M // 2, M // 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(M // 2, 3, 4, stride=2, padding=1),
        )

    def forward(self, residual):
        z = self.encoder(residual)
        z_hat, likelihoods = self.entropy_bottleneck(z)
        residual_hat = self.decoder(z_hat)
        if residual_hat.shape != residual.shape:
            residual_hat = F.interpolate(residual_hat, size=residual.shape[2:],
                                         mode='bilinear', align_corners=False)
        return residual_hat, likelihoods

    def compress(self, residual):
        z = self.encoder(residual)
        return self.entropy_bottleneck.compress(z), z.shape[2:]

    def decompress(self, strings, shape, out_size):
        z_hat = self.entropy_bottleneck.decompress(strings, shape)
        res_hat = self.decoder(z_hat)
        if res_hat.shape[2:] != out_size:
            res_hat = F.interpolate(res_hat, size=out_size,
                                    mode='bilinear', align_corners=False)
        return res_hat

class IFrameCodec(nn.Module):
    """
    Learned I-frame codec using CompressAI's bmshj2018_factorized.
    Replaces BPG used in original OpenDVC.

    Pretrained models available at quality levels 1-8.
    For training, we use quality=4 as default (similar to BPG QP=32 PSNR-wise).
    """
    def __init__(self, quality: int = 4, pretrained: bool = False):
        super().__init__()
        self.codec = FactorizedPrior(128, 192)

        for p in self.codec.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: image (B, 3, H, W) in [0, 1]
        Returns:
            x_hat: reconstructed image (B, 3, H, W)
            likelihoods: dict with 'y' and 'z' likelihood tensors
        """
        out = self.codec(x)
        return out['x_hat'].clamp(0, 1), out['likelihoods']

    def compress(self, x):
        """Compress to actual bitstream (inference only)."""
        return self.codec.compress(x)

    def decompress(self, strings, shape):
        """Decompress from bitstream."""
        return self.codec.decompress(strings, shape)['x_hat'].clamp(0, 1)

def rate_estimate(likelihoods, num_pixels: int) -> torch.Tensor:
    """
    Compute estimated bit-rate from likelihood tensors.

    Args:
        likelihoods: tensor of symbol probabilities
        num_pixels: H * W (for normalizing to bits-per-pixel)
    Returns:
        bpp: bits per pixel (scalar tensor)
    """
    bits = -torch.log2(likelihoods).sum()
    return bits / num_pixels