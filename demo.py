"""
demo.py — Quick demo: run full pipeline on one frame pair or a video clip.

Chạy trong VS Code terminal:

    python demo.py

    python demo.py --ref path/to/frame1.png --cur path/to/frame2.png

    python demo.py --video path/to/video.mp4 --start_frame 10

    python demo.py --checkpoint checkpoints/psnr_l512/best.pth
"""

import os, sys, argparse
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.dvc_model import DVCModel
from models.flow_net import bilinear_warp
from utils.metrics import psnr, ssim, residual_entropy
from utils.visualization import make_comparison_grid

def get_args():
    p = argparse.ArgumentParser(description='Learned Motion Compensation Demo')
    p.add_argument('--checkpoint', type=str,  default=None)
    p.add_argument('--ref',        type=str,  default=None, help='Reference frame PNG')
    p.add_argument('--cur',        type=str,  default=None, help='Current frame PNG')
    p.add_argument('--video',      type=str,  default=None, help='Video file path')
    p.add_argument('--start_frame',type=int,  default=0,    help='Start frame index for video')
    p.add_argument('--frames',     type=int,  default=10,   help='Number of consecutive frames to predict')
    p.add_argument('--height',     type=int,  default=256)
    p.add_argument('--width',      type=int,  default=256)
    p.add_argument('--lmbda',      type=int,  default=512)
    p.add_argument('--flow_M',     type=int,  default=128)
    p.add_argument('--res_M',      type=int,  default=128)
    p.add_argument('--output',     type=str,  default='results/demo_output.png')
    p.add_argument('--device',     type=str,  default='auto')
    return p.parse_args()

def load_model(args, device):

    flow_M, res_M, lmbda = args.flow_M, args.res_M, args.lmbda
    ckpt = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg  = ckpt.get('args', {})
        flow_M = cfg.get('flow_M', flow_M)
        res_M  = cfg.get('res_M',  res_M)
        lmbda  = cfg.get('lmbda',  lmbda)

    model = DVCModel(flow_M=flow_M, res_M=res_M, lmbda=lmbda)

    if ckpt is not None:
        model.load_state_dict(ckpt['model'], strict=True)
        cfg = ckpt.get('args', {})
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"  Config: λ={cfg.get('lmbda','?')}  metric={cfg.get('metric','?')}  flow_M={flow_M}  res_M={res_M}")
    else:
        print("No checkpoint — using random weights (structure demo only)")
    model.to(device).eval()
    return model

def load_png(path, height, width):
    """Load PNG → (1,3,H,W) float tensor [0,1]."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))
    t   = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t

def load_video_sequence(video_path, start_frame, num_frames, height, width):
    """Extract consecutive frames from a video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(num_frames):
        ret, f = cap.read()
        if not ret:
            break
        f = cv2.cvtColor(cv2.resize(f, (width, height)), cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(f.astype(np.float32) / 255.0)
                         .permute(2, 0, 1).unsqueeze(0))
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"Could not read enough frames from {video_path}")
    return frames

def run_pipeline(model, ref_t, cur_t, device):
    """Full encode-decode pipeline, return all intermediate results."""
    ref_t = ref_t.to(device)
    cur_t = cur_t.to(device)

    with torch.no_grad():
        frame_rec, losses = model(ref_t, cur_t)
        flow = model.flow_net(ref_t, cur_t)

        flow_hat, _ = model.motion_coder(flow)
        pred = bilinear_warp(ref_t, flow_hat)
        residual = cur_t - pred

    metrics = {
        'psnr':       psnr(cur_t, frame_rec),
        'ssim':       ssim(cur_t, frame_rec),
        'bpp':        losses['bpp'].item(),
        'R_motion':   losses['R_motion'].item(),
        'R_residual': losses['R_residual'].item(),
        'entropy':    residual_entropy(residual[0].cpu()),
    }

    print("\n── Pipeline Results ──────────────────────────────")
    print(f"  PSNR (reconstructed vs original) : {metrics['psnr']:.2f} dB")
    print(f"  SSIM                             : {metrics['ssim']:.4f}")
    print(f"  bpp (motion + residual)          : {metrics['bpp']:.5f}")
    print(f"  R_motion                         : {metrics['R_motion']:.5f} bpp")
    print(f"  R_residual                       : {metrics['R_residual']:.5f} bpp")
    print(f"  Residual entropy                 : {metrics['entropy']:.3f} bits")
    print("──────────────────────────────────────────────────\n")

    return {
        'ref':        ref_t,
        'cur':        cur_t,
        'pred':       pred,
        'frame_rec':  frame_rec,
        'flow':       flow_hat,
        'residual':   residual,
        'metrics':    metrics,
    }

def main():
    args   = get_args()
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))
    print(f"Device: {device}")

    model = load_model(args, device)
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    if args.video:
        print(f"Loading {args.frames + 1} frames from video: {args.video}  (start frame {args.start_frame})")
        frames = load_video_sequence(args.video, args.start_frame, args.frames + 1, args.height, args.width)
        
        ref_t = frames[0]
        base_out = args.output
        name, ext = os.path.splitext(base_out)
        
        for i in range(1, len(frames)):
            cur_t = frames[i]
            print(f"\n=== Predicting Frame {i}/{len(frames)-1} ===")
            results = run_pipeline(model, ref_t, cur_t, device)
            
            out_path = f"{name}_{i:02d}{ext}"
            make_comparison_grid(
                results['ref'],   results['cur'],
                results['pred'],  results['frame_rec'],
                results['flow'],  results['residual'],
                results['metrics'],
                save_path=out_path,
            )
            print(f"Saved: {out_path}")

            ref_t = results['frame_rec'].detach()
            
    else:
        if args.ref and args.cur:
            print(f"Loading frames: {args.ref}  →  {args.cur}")
            ref_t = load_png(args.ref, args.height, args.width)
            cur_t = load_png(args.cur, args.height, args.width)
        else:
            raise SystemExit(
                "Provide input: --video <file>, or --ref <png> and --cur <png>.")

        results = run_pipeline(model, ref_t, cur_t, device)
        make_comparison_grid(
            results['ref'],   results['cur'],
            results['pred'],  results['frame_rec'],
            results['flow'],  results['residual'],
            results['metrics'],
            save_path=args.output,
        )
        print(f"Demo figure saved: {args.output}")

if __name__ == '__main__':
    main()