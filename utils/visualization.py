"""Chuyển tensor -> ảnh hiển thị cho app demo."""

import numpy as np
import torch
import cv2


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(B,C,H,W) hoặc (C,H,W) float [0,1] -> (H,W,3) uint8."""
    if t.dim() == 4:
        t = t[0]
    img = t.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def flow_to_color(flow: torch.Tensor) -> np.ndarray:
    """Flow (2,H,W) -> ảnh màu: hue = hướng, độ sáng = độ lớn chuyển động."""
    if flow.dim() == 4:
        flow = flow[0]
    flow_np = flow.detach().cpu().numpy()
    u, v = flow_np[0], flow_np[1]
    mag = np.sqrt(u ** 2 + v ** 2)
    ang = np.arctan2(v, u)
    hue = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    sat = np.full_like(hue, 255, dtype=np.uint8)
    val = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)
    hsv = np.stack([hue, sat, val], axis=2)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
