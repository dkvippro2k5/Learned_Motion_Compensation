"""
tests/test_smoke.py — fast, offline sanity checks for the core pipeline.

Runs on CPU with tiny tensors (no dataset, no network, no checkpoint). Verifies
that the model trains (forward + backward), produces finite rate/distortion, and
that the real entropy-coding roundtrip (compress -> decompress) is consistent.

Run directly:
    python tests/test_smoke.py
or with pytest:
    pytest tests/test_smoke.py
"""

import os
import sys
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dvc_model import DVCModel
from models.flow_net import PWCNet, bilinear_warp
from utils.metrics import psnr, ssim


def _frames(h=64, w=64, b=2):
    torch.manual_seed(0)
    return torch.rand(b, 3, h, w), torch.rand(b, 3, h, w)


def test_flow_and_warp_shapes():
    ref, cur = _frames()
    flow = PWCNet()(ref, cur)
    assert flow.shape == (ref.shape[0], 2, ref.shape[2], ref.shape[3])
    warped = bilinear_warp(ref, flow)
    assert warped.shape == ref.shape
    print("ok  flow + warp shapes")


def test_forward_backward_finite():
    ref, cur = _frames()
    model = DVCModel(flow_M=64, res_M=64, lmbda=1024)
    rec, losses = model(ref, cur)
    assert rec.shape == cur.shape
    assert 0.0 <= rec.min() and rec.max() <= 1.0
    for k in ('loss_psnr', 'bpp', 'R_motion', 'R_residual', 'D_mse'):
        v = losses[k]
        assert torch.isfinite(v).all(), f"{k} not finite"
    assert losses['bpp'].item() > 0
    losses['loss_psnr'].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0, "no gradients flowed"
    print(f"ok  forward+backward (bpp={losses['bpp'].item():.3f})")


def test_entropy_roundtrip_consistent():
    ref, cur = _frames()
    model = DVCModel(flow_M=64, res_M=64, lmbda=1024).eval()
    model.update(force=True)
    with torch.no_grad():
        bitstream, frame_rec = model.encode_frame(ref[:1], cur[:1])
        decoded = model.decode_frame(ref[:1], bitstream)
    assert decoded.shape == cur[:1].shape
    # encoder-side and decoder-side reconstruction must match (same bitstream).
    max_diff = (frame_rec - decoded).abs().max().item()
    assert max_diff < 1e-3, f"roundtrip mismatch {max_diff}"
    print(f"ok  entropy roundtrip (max|enc-dec|={max_diff:.2e})")


def test_metrics_sane():
    _, cur = _frames()
    assert math.isinf(psnr(cur, cur))           # identical -> inf PSNR
    assert abs(ssim(cur, cur) - 1.0) < 1e-4      # identical -> SSIM 1
    print("ok  metrics sane")


def main():
    tests = [
        test_flow_and_warp_shapes,
        test_forward_backward_finite,
        test_entropy_roundtrip_consistent,
        test_metrics_sane,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} smoke tests passed.")


if __name__ == '__main__':
    main()
