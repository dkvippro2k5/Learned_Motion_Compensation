# Full DVC-inspired Learned Video Compression model.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .flow_net import PWCNet, bilinear_warp
from .entropy_coder import MotionCompressor, ResidualCompressor, rate_estimate

class DVCModel(nn.Module):

    def __init__(self, flow_M=128, res_M=128, lmbda=512):
        super().__init__()
        self.lmbda = lmbda
        self.flow_net = PWCNet(max_disp=4)
        self.motion_coder = MotionCompressor(M=flow_M)
        self.residual_coder = ResidualCompressor(M=res_M)

    def forward(self, frame_ref, frame_cur):
        
        B, C, H, W = frame_cur.shape
        num_pixels = B * H * W

        flow_raw = self.flow_net(frame_ref, frame_cur)

        flow_hat, motion_likelihoods = self.motion_coder(flow_raw)

        frame_pred = bilinear_warp(frame_ref, flow_hat)

        residual = frame_cur - frame_pred

        residual_hat, res_likelihoods = self.residual_coder(residual)

        frame_rec = (frame_pred + residual_hat).clamp(0, 1)

        R_motion   = rate_estimate(motion_likelihoods, num_pixels)
        R_residual = rate_estimate(res_likelihoods,    num_pixels)
        R_total    = R_motion + R_residual

        D_mse = F.mse_loss(frame_rec, frame_cur)

        D_msssim = 1.0 - ms_ssim(frame_rec, frame_cur)

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
    def encode_frame(self, frame_ref, frame_cur, return_intermediates=False):
        """Nén THẬT một P-frame (arithmetic coding flow + residual) rồi giải mã lại.

        Trả về (bitstream_dict, frame_rec). Nếu return_intermediates=True, trả thêm
        dict {flow_hat, pred, residual} để demo (app.py) vẽ motion field / residual
        mà không phải nén lại. Đây là nơi DUY NHẤT chứa logic nén thật của codec.
        """
        B, C, H, W = frame_cur.shape

        flow_raw = self.flow_net(frame_ref, frame_cur)
        motion_strings, motion_shape = self.motion_coder.compress(flow_raw)
        flow_hat = self.motion_coder.decompress(motion_strings, motion_shape, (H, W))

        frame_pred = bilinear_warp(frame_ref, flow_hat)
        residual = frame_cur - frame_pred

        res_strings, res_shape = self.residual_coder.compress(residual)
        residual_hat = self.residual_coder.decompress(res_strings, res_shape, (H, W))
        frame_rec = (frame_pred + residual_hat).clamp(0, 1)

        bitstream = {
            'motion_strings':  motion_strings,
            'residual_strings': res_strings,
            'motion_shape':    motion_shape,
            'res_shape':       res_shape,
            'H': H, 'W': W,
        }
        if return_intermediates:
            return bitstream, frame_rec, {
                'flow_hat': flow_hat, 'pred': frame_pred, 'residual': residual}
        return bitstream, frame_rec

    def update(self, force: bool = True):
        """Build entropy-coder CDF tables. Call once before compress/decompress."""
        self.motion_coder.update(force=force)
        self.residual_coder.update(force=force)

def _gaussian_window(size=11, sigma=1.5, channels=3):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    return kernel.expand(channels, 1, size, size)

def ms_ssim(img1, img2, levels=3, window_size=11):

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

        if i < levels - 1:
            img1 = F.avg_pool2d(img1, 2)
            img2 = F.avg_pool2d(img2, 2)

    val = torch.stack(mssim)
    return (val * weights).sum()