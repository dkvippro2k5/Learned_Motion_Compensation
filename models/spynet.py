"""
SpyNet - Spatial Pyramid Network for Optical Flow
Reference: Ranjan & Black, CVPR 2017
Simplified implementation for Learned Motion Compensation project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModule(nn.Module):
    """Basic convolutional module used in each pyramid level of SpyNet."""
    def __init__(self):
        super(BasicModule, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, x):
        return self.net(x)


class SpyNet(nn.Module):
    """
    SpyNet: Spatial Pyramid Network for Optical Flow Estimation.
    
    Input:  Two consecutive frames I_{t-1} and I_t  (B, 3, H, W) each
    Output: Dense optical flow field                 (B, 2, H, W)
            - channel 0: horizontal displacement (u)
            - channel 1: vertical displacement   (v)
    """
    def __init__(self, num_levels=4):
        super(SpyNet, self).__init__()
        self.num_levels = num_levels
        self.basic_modules = nn.ModuleList([BasicModule() for _ in range(num_levels)])
        
        # Mean and std for normalization (ImageNet stats)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def warp(self, x, flow):
        """Bilinear warping using a flow field."""
        B, C, H, W = x.size()
        # Normalize grid to [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=x.device),
            torch.arange(W, dtype=torch.float32, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # (1,2,H,W)
        
        # Apply flow
        new_grid = grid + flow
        new_grid[:, 0] = 2.0 * new_grid[:, 0] / max(W - 1, 1) - 1.0
        new_grid[:, 1] = 2.0 * new_grid[:, 1] / max(H - 1, 1) - 1.0
        new_grid = new_grid.permute(0, 2, 3, 1)  # (B,H,W,2)
        
        return F.grid_sample(x, new_grid, mode='bilinear', padding_mode='border', align_corners=True)

    def forward(self, frame_ref, frame_cur):
        """
        Args:
            frame_ref: Reference frame I_{t-1}  (B, 3, H, W), values in [0, 1]
            frame_cur: Current frame   I_t       (B, 3, H, W), values in [0, 1]
        Returns:
            flow: Dense optical flow (B, 2, H, W)
        """
        I_ref = self.normalize(frame_ref)
        I_cur = self.normalize(frame_cur)

        # Build image pyramids
        pyramid_ref = [I_ref]
        pyramid_cur = [I_cur]
        for _ in range(self.num_levels - 1):
            pyramid_ref.append(F.avg_pool2d(pyramid_ref[-1], 2))
            pyramid_cur.append(F.avg_pool2d(pyramid_cur[-1], 2))

        # Coarse-to-fine flow estimation
        flow = torch.zeros(
            frame_ref.size(0), 2,
            pyramid_ref[-1].size(2),
            pyramid_ref[-1].size(3),
            device=frame_ref.device
        )

        for level in reversed(range(self.num_levels)):
            ref_l = pyramid_ref[level]
            cur_l = pyramid_cur[level]
            
            # Upsample flow from coarser level
            if level < self.num_levels - 1:
                flow = F.interpolate(flow, size=ref_l.shape[2:], mode='bilinear', align_corners=True) * 2.0

            # Warp reference frame using current flow estimate
            ref_warped = self.warp(ref_l, flow)

            # Concatenate: warped_ref, cur, flow => 8 channels
            inp = torch.cat([ref_warped, cur_l, flow], dim=1)  # (B, 8, H, W)

            # Predict residual flow correction
            delta_flow = self.basic_modules[level](inp)
            flow = flow + delta_flow

        return flow


class LightweightFlowNet(nn.Module):
    """
    A lightweight custom CNN for optical flow estimation.
    More suitable for limited hardware. Single-scale prediction.
    
    Input:  Concatenation of I_{t-1} and I_t  (B, 6, H, W)
    Output: Dense optical flow field           (B, 2, H, W)
    """
    def __init__(self):
        super(LightweightFlowNet, self).__init__()
        
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(6, 32, 7, padding=3), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            # Block 2
            nn.Conv2d(32, 64, 5, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),
        )

    def forward(self, frame_ref, frame_cur):
        x = torch.cat([frame_ref, frame_cur], dim=1)
        features = self.encoder(x)
        flow = self.decoder(features)
        return flow