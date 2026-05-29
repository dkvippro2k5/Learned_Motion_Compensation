"""
models/flow_net.py
PWC-Net inspired optical flow network — PyTorch 2.x, CPU/GPU compatible.

Reference: Sun et al., "PWC-Net: CNNs for Optical Flow Using Pyramid,
           Warping, and Cost Volume", CVPR 2018.

Key improvements over LightweightFlowNet (v1):
- Cost volume layer (cross-correlation) for more accurate matching
- Coarse-to-fine estimation (pyramid)
- Explicit context network for refinement
- ~1.4M params — still lightweight for CPU training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_relu(in_ch, out_ch, kernel=3, stride=1, dilation=1):
    pad = dilation * (kernel // 2)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                  padding=pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class FeatureExtractor(nn.Module):
    """Extract multi-scale features from a single frame."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(conv_bn_relu(3,  16, 3, 2), conv_bn_relu(16, 16, 3, 1))
        self.layer2 = nn.Sequential(conv_bn_relu(16, 32, 3, 2), conv_bn_relu(32, 32, 3, 1))
        self.layer3 = nn.Sequential(conv_bn_relu(32, 64, 3, 2), conv_bn_relu(64, 64, 3, 1))
        self.layer4 = nn.Sequential(conv_bn_relu(64, 96, 3, 2), conv_bn_relu(96, 96, 3, 1))

    def forward(self, x):
        f1 = self.layer1(x)   # /2
        f2 = self.layer2(f1)  # /4
        f3 = self.layer3(f2)  # /8
        f4 = self.layer4(f3)  # /16
        return [f1, f2, f3, f4]


class CostVolume(nn.Module):
    """
    Compute cross-correlation cost volume between two feature maps.
    For each displacement d in [-D, D], compute dot product.
    """
    def __init__(self, max_disp=4):
        super().__init__()
        self.max_disp = max_disp

    def forward(self, feat1, feat2):
        B, C, H, W = feat1.shape
        D = self.max_disp
        cost = []
        feat2_pad = F.pad(feat2, [D, D, D, D])
        for dy in range(-D, D + 1):
            for dx in range(-D, D + 1):
                f2_shift = feat2_pad[:, :,
                                     D + dy: D + dy + H,
                                     D + dx: D + dx + W]
                corr = (feat1 * f2_shift).mean(dim=1, keepdim=True)
                cost.append(corr)
        return torch.cat(cost, dim=1)  # (B, (2D+1)^2, H, W)


class FlowDecoder(nn.Module):
    """
    Decode flow from cost volume + features + upsampled coarser flow.
    in_channels = cost_vol_channels + feature_channels + 2 (upsampled flow)
    """
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            conv_bn_relu(in_ch,      128, 3),
            conv_bn_relu(128,         96, 3),
            conv_bn_relu(96,           64, 3),
            conv_bn_relu(64,           32, 3),
        )
        self.flow_pred = nn.Conv2d(32, 2, 3, padding=1)
        nn.init.zeros_(self.flow_pred.weight)
        nn.init.zeros_(self.flow_pred.bias)

    def forward(self, x):
        feat = self.conv(x)
        return self.flow_pred(feat)


class ContextNet(nn.Module):
    """
    Context / refinement network: dilated convolutions for larger receptive field.
    Takes concatenation of features + coarse flow as input.
    """
    def __init__(self, in_ch=34):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn_relu(in_ch, 128, 3, dilation=1),
            conv_bn_relu(128,    128, 3, dilation=2),
            conv_bn_relu(128,    128, 3, dilation=4),
            conv_bn_relu(128,     96, 3, dilation=8),
            conv_bn_relu(96,      64, 3, dilation=16),
            conv_bn_relu(64,      32, 3, dilation=1),
            nn.Conv2d(32, 2, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class PWCNet(nn.Module):
    """
    PWC-Net: Pyramid, Warping, Cost volume network.

    Input:  frame_ref (B, 3, H, W),  frame_cur (B, 3, H, W)   — values in [0, 1]
    Output: flow (B, 2, H, W)                                   — pixel displacements

    Architecture:
        1. Extract 4-level feature pyramids from both frames
        2. At each level (coarse→fine):
           a. Warp ref features using upsampled flow
           b. Compute cost volume between warped ref and cur features
           c. Decode flow from [cost_vol | cur_feat | upsampled_flow]
        3. Final context network refines the full-resolution flow
    """
    def __init__(self, max_disp=4):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.cost_volume = CostVolume(max_disp)
        cv_ch = (2 * max_disp + 1) ** 2  # 81 channels

        # Decoders for each pyramid level (coarsest first = level 4)
        self.decoder4 = FlowDecoder(cv_ch + 96 + 0)   # no upsampled flow at coarsest
        self.decoder3 = FlowDecoder(cv_ch + 64 + 2)
        self.decoder2 = FlowDecoder(cv_ch + 32 + 2)

        # Context network on level-2 features
        self.context = ContextNet(in_ch=32 + 2)

    @staticmethod
    def warp(feat, flow):
        """Bilinear warp of a feature map using a flow field."""
        B, C, H, W = feat.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=feat.device),
            torch.arange(W, dtype=torch.float32, device=feat.device),
            indexing='ij',
        )
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        new_grid = grid + flow
        new_grid[:, 0] = 2.0 * new_grid[:, 0] / max(W - 1, 1) - 1.0
        new_grid[:, 1] = 2.0 * new_grid[:, 1] / max(H - 1, 1) - 1.0
        return F.grid_sample(feat, new_grid.permute(0, 2, 3, 1),
                             mode='bilinear', padding_mode='border', align_corners=True)

    def forward(self, frame_ref, frame_cur):
        # Extract feature pyramids
        feats_ref = self.feature_extractor(frame_ref)  # [f1..f4]
        feats_cur = self.feature_extractor(frame_cur)

        # Level 4 (coarsest, 1/16 resolution) — no initial flow
        cv4 = self.cost_volume(feats_ref[3], feats_cur[3])
        flow4 = self.decoder4(torch.cat([cv4, feats_cur[3]], dim=1))

        # Level 3 (1/8)
        up_flow4 = F.interpolate(flow4, scale_factor=2, mode='bilinear', align_corners=True) * 2
        warped3 = self.warp(feats_ref[2], up_flow4)
        cv3 = self.cost_volume(warped3, feats_cur[2])
        flow3 = self.decoder3(torch.cat([cv3, feats_cur[2], up_flow4], dim=1)) + up_flow4

        # Level 2 (1/4)
        up_flow3 = F.interpolate(flow3, scale_factor=2, mode='bilinear', align_corners=True) * 2
        warped2 = self.warp(feats_ref[1], up_flow3)
        cv2 = self.cost_volume(warped2, feats_cur[1])
        flow2 = self.decoder2(torch.cat([cv2, feats_cur[1], up_flow3], dim=1)) + up_flow3

        # Context refinement at level 2
        flow2_refined = flow2 + self.context(torch.cat([feats_cur[1], flow2], dim=1))

        # Upsample to full resolution
        flow_full = F.interpolate(flow2_refined, scale_factor=4,
                                  mode='bilinear', align_corners=True) * 4

        return flow_full


# ── Bilinear warp (frame-level, not feature-level) ──────────────────────────

def bilinear_warp(frame, flow):
    """
    Warp a frame (B, C, H, W) using dense flow (B, 2, H, W).
    Suitable for both frames and feature maps.
    """
    B, C, H, W = frame.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=frame.device),
        torch.arange(W, dtype=torch.float32, device=frame.device),
        indexing='ij',
    )
    grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    new_grid = grid + flow
    new_grid[:, 0] = 2.0 * new_grid[:, 0] / max(W - 1, 1) - 1.0
    new_grid[:, 1] = 2.0 * new_grid[:, 1] / max(H - 1, 1) - 1.0
    return F.grid_sample(frame, new_grid.permute(0, 2, 3, 1),
                         mode='bilinear', padding_mode='border', align_corners=True)