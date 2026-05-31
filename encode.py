# P-frame encoder (like OpenDVC_test_P-frame.py but PyTorch 2.x)


import os, sys, argparse, pickle, struct, math
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.dvc_model import DVCModel
from utils.metrics import psnr, ssim, ms_ssim_numpy, bitstream_bpp

def get_args():
    p = argparse.ArgumentParser(description='DVC-inspired P-frame encoder')

    p.add_argument('--ref',    type=str, default=None, help='Reference frame PNG')
    p.add_argument('--raw',    type=str, default=None, help='Current frame PNG to compress')
    p.add_argument('--com',    type=str, default=None, help='Compressed frame output PNG')
    p.add_argument('--bin',    type=str, default=None, help='Bitstream output path')

    p.add_argument('--path',   type=str, default=None, help='Path to PNG sequence folder')
    p.add_argument('--frame',  type=int, default=100,  help='Number of frames to encode')
    p.add_argument('--GOP',    type=int, default=10,   help='GOP size')

    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--lmbda',      type=int, default=1024)
    p.add_argument('--metric',     type=str, default='psnr', choices=['psnr', 'msssim'])
    p.add_argument('--flow_M',     type=int, default=128)
    p.add_argument('--res_M',      type=int, default=128)

    p.add_argument('--height', type=int, default=None)
    p.add_argument('--width',  type=int, default=None)
    p.add_argument('--device', type=str, default='auto')

    return p.parse_args()

def load_model(args, device):
    model = DVCModel(flow_M=args.flow_M, res_M=args.res_M, lmbda=args.lmbda)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        print(f"Loaded: {args.checkpoint}")
    else:
        print("Warning: No checkpoint — using random weights (demo only)")
    model.to(device).eval()
    return model

def load_frame(path, height=None, width=None):
    """Load PNG as (1, 3, H, W) float tensor in [0, 1]."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    if height and width:
        img = cv2.resize(img, (width, height))
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t

def save_frame(tensor, path):

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if tensor.dim() == 4: tensor = tensor[0]
    img = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def save_bitstream(bitstream_dict, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        data = pickle.dumps(bitstream_dict)
        f.write(struct.pack('>I', len(data)))
        f.write(data)

def load_bitstream(path):
    """Deserialize bitstream dict from binary file."""
    with open(path, 'rb') as f:
        size = struct.unpack('>I', f.read(4))[0]
        return pickle.loads(f.read(size))

def encode_one_pframe(model, ref_tensor, cur_tensor, bin_path, com_path, device):
    """Encode one P-frame. Returns dict with metrics."""
    ref = ref_tensor.to(device)
    cur = cur_tensor.to(device)
    H, W = cur.shape[2], cur.shape[3]

    with torch.no_grad():
        bitstream_dict, frame_rec = model.encode_frame(ref, cur)

    if com_path:
        save_frame(frame_rec, com_path)

    if bin_path:
        save_bitstream(bitstream_dict, bin_path)
        file_bytes = os.path.getsize(bin_path)
        bpp = file_bytes * 8 / (H * W)
    else:

        motion_bytes = sum(len(s[0]) for s in [bitstream_dict['motion_strings']] if s)
        bpp = (motion_bytes) * 8 / (H * W)

    metrics = {
        'psnr': psnr(cur, frame_rec),
        'ssim': ssim(cur, frame_rec),
        'bpp':  bpp,
        'H': H, 'W': W,
    }
    return frame_rec, metrics

def encode_video(model, frame_dir, out_com_dir, out_bin_dir, n_frames, gop, device):

    os.makedirs(out_com_dir, exist_ok=True)
    os.makedirs(out_bin_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])[:n_frames]
    if not frame_files:
        raise FileNotFoundError(f"No PNG files in {frame_dir}")

    all_metrics = []
    ref_tensor = None

    for i, fname in enumerate(frame_files):
        src_path = os.path.join(frame_dir, fname)
        com_path = os.path.join(out_com_dir, fname)
        bin_path = os.path.join(out_bin_dir, fname.replace('.png', '.bin'))
        frame_num = i + 1

        cur_tensor = load_frame(src_path)
        H, W = cur_tensor.shape[2], cur_tensor.shape[3]

        if i % gop == 0:

            import shutil
            shutil.copy2(src_path, com_path)
            ref_tensor = cur_tensor.clone()
            metrics = {
                'frame': frame_num,
                'type': 'I',
                'psnr': float('inf'),
                'ssim': 1.0,
                'bpp': H * W * 24 / (H * W),  # raw RGB as placeholder
            }
            print(f"  Frame {frame_num:3d} [I] PSNR:  inf    SSIM: 1.0000  bpp: 24.00")
        else:

            frame_rec, metrics = encode_one_pframe(
                model, ref_tensor, cur_tensor, bin_path, com_path, device)
            metrics['frame'] = frame_num
            metrics['type']  = 'P'
            ref_tensor = frame_rec.cpu()
            print(f"  Frame {frame_num:3d} [P] "
                  f"PSNR:{metrics['psnr']:6.2f}  "
                  f"SSIM:{metrics['ssim']:.4f}  "
                  f"bpp:{metrics['bpp']:.4f}")

        all_metrics.append(metrics)

    return all_metrics

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
             if args.device == 'auto' else torch.device(args.device)

    model = load_model(args, device)

    if args.ref and args.raw:
        print(f"Encoding single P-frame: {args.raw}")
        ref_t = load_frame(args.ref, args.height, args.width)
        cur_t = load_frame(args.raw, args.height, args.width)

        _, metrics = encode_one_pframe(
            model, ref_t, cur_t, args.bin, args.com, device)

        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  bpp:  {metrics['bpp']:.5f}")
        if args.com: print(f"  Reconstructed frame: {args.com}")
        if args.bin: print(f"  Bitstream: {args.bin}")

    elif args.path:
        mode_str = f"{args.metric.upper()}_l{args.lmbda}"
        out_com = args.path + f'_com_{mode_str}'
        out_bin = args.path + f'_bin_{mode_str}'

        print(f"Encoding video: {args.path}")
        print(f"  Frames: {args.frame}  GOP: {args.GOP}")

        all_metrics = encode_video(
            model, args.path, out_com, out_bin,
            args.frame, args.GOP, device)

        p_metrics = [m for m in all_metrics if m['type'] == 'P']
        if p_metrics:
            avg_psnr = np.mean([m['psnr'] for m in p_metrics])
            avg_ssim = np.mean([m['ssim'] for m in p_metrics])
            avg_bpp  = np.mean([m['bpp']  for m in p_metrics])
            print(f"\nP-frame average:")
            print(f"  PSNR: {avg_psnr:.2f} dB")
            print(f"  SSIM: {avg_ssim:.4f}")
            print(f"  bpp:  {avg_bpp:.5f}")
            print(f"\nOutput: {out_com}/  (frames)")
            print(f"        {out_bin}/   (bitstreams)")

    else:
        print("Usage: python encode.py --ref ref.png --raw cur.png --com out.png --bin out.bin")
        print("   or: python encode.py --path frames/ --frame 100 --GOP 10")

if __name__ == '__main__':
    main()