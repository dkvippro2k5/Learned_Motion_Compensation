# Entropy coding for motion fields and residuals using CompressAI.
#
# Both the motion field and the residual are compressed with a SCALE HYPERPRIOR
# (Ballé et al., 2018): a hyper-network predicts per-element scales for a
# Gaussian conditional entropy model on the main latent. This is a large
# upgrade over a plain factorized EntropyBottleneck — the scales let the model
# spend bits only where the latent is unpredictable, which is what drives the
# bitrate (bpp) down. The main transform also downsamples by /16 (vs /8 before)
# so the latent has far fewer elements than the image.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import GDN
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


def _conv(in_ch, out_ch, k=5, s=2):
    return nn.Conv2d(in_ch, out_ch, k, stride=s, padding=k // 2)


def _deconv(in_ch, out_ch, k=5, s=2):
    return nn.ConvTranspose2d(in_ch, out_ch, k, stride=s,
                              padding=k // 2, output_padding=s - 1)


def get_scale_table(smin=0.11, smax=256.0, levels=64):
    """Geometric scale table for the GaussianConditional (CompressAI default)."""
    return torch.exp(torch.linspace(math.log(smin), math.log(smax), levels))


class HyperpriorCompressor(nn.Module):
    """Generic scale-hyperprior codec for an arbitrary-channel signal.

    g_a: x -> y      (main encoder, /16)
    h_a: |y| -> z    (hyper encoder)
    z  -> z_hat      (EntropyBottleneck, factorized prior)
    h_s: z_hat -> scales
    GaussianConditional(y | scales) -> y_hat
    g_s: y_hat -> x_hat
    """

    def __init__(self, in_ch: int, N: int = 128, M: int = 192):
        super().__init__()
        self.in_ch = in_ch
        self.N, self.M = N, M

        self.g_a = nn.Sequential(
            _conv(in_ch, N), GDN(N),
            _conv(N, N),     GDN(N),
            _conv(N, N),     GDN(N),
            _conv(N, M),
        )
        self.g_s = nn.Sequential(
            _deconv(M, N), GDN(N, inverse=True),
            _deconv(N, N), GDN(N, inverse=True),
            _deconv(N, N), GDN(N, inverse=True),
            _deconv(N, in_ch),
        )
        self.h_a = nn.Sequential(
            _conv(M, N, k=3, s=1), nn.LeakyReLU(0.1, inplace=True),
            _conv(N, N),           nn.LeakyReLU(0.1, inplace=True),
            _conv(N, N),
        )
        self.h_s = nn.Sequential(
            _deconv(N, N), nn.LeakyReLU(0.1, inplace=True),
            _deconv(N, N), nn.LeakyReLU(0.1, inplace=True),
            _conv(N, M, k=3, s=1), nn.LeakyReLU(0.1, inplace=True),
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)
        if x_hat.shape[2:] != x.shape[2:]:
            x_hat = F.interpolate(x_hat, size=x.shape[2:],
                                  mode='bilinear', align_corners=False)
        return x_hat, {'y': y_likelihoods, 'z': z_likelihoods}

    def update(self, force: bool = True):
        """Build the CDF tables needed for real entropy coding (compress)."""
        updated = self.gaussian_conditional.update_scale_table(
            get_scale_table().to(self._device()), force=force)
        updated |= self.entropy_bottleneck.update(force=force)
        return updated

    def _device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return [y_strings, z_strings], z.size()[-2:]

    @torch.no_grad()
    def decompress(self, strings, z_shape, out_size):
        y_strings, z_strings = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(y_strings, indexes)
        x_hat = self.g_s(y_hat)
        if out_size is not None and x_hat.shape[2:] != tuple(out_size):
            x_hat = F.interpolate(x_hat, size=out_size,
                                  mode='bilinear', align_corners=False)
        return x_hat


class MotionCompressor(HyperpriorCompressor):
    """Scale-hyperprior codec for the 2-channel optical-flow field."""
    def __init__(self, M: int = 128):
        # M here is treated as the main-latent width (kept name for CLI compat).
        super().__init__(in_ch=2, N=128, M=M)


class ResidualCompressor(HyperpriorCompressor):
    """Scale-hyperprior codec for the 3-channel prediction residual."""
    def __init__(self, M: int = 128):
        super().__init__(in_ch=3, N=128, M=M)


def rate_estimate(likelihoods, num_pixels: int) -> torch.Tensor:
    """Estimated bitrate (bpp). Accepts a single likelihood tensor or a dict of
    likelihood tensors (hyperprior returns {'y':..., 'z':...})."""
    if isinstance(likelihoods, dict):
        bits = sum(-torch.log2(l.clamp(min=1e-9)).sum() for l in likelihoods.values())
    else:
        bits = -torch.log2(likelihoods.clamp(min=1e-9)).sum()
    return bits / num_pixels
