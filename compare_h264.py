"""
compare_h264.py — Honest Rate-Distortion comparison: learned DVC model vs H.264.

Both codecs are evaluated on the SAME frame sequence:
  * H.264 via ffmpeg/libx264 at several CRF values (real bitstream size -> bpp).
  * The learned DVC model by running its actual forward pass and reading the
    estimated rate (bpp) and reconstruction quality (PSNR/SSIM).

DVC numbers are NOT hardcoded — pass one or more trained checkpoints with
--checkpoints. Each checkpoint (a given lambda) becomes one point on the DVC
R-D curve. If no checkpoint is given, only the H.264 baseline is plotted.

Usage:
    python compare_h264.py \
        --input_dir data/real_test/cur \
        --frame_pattern "%05d.png" \
        --checkpoints checkpoints/psnr_l256/best.pth checkpoints/psnr_l1024/best.pth
"""

import os
import sys
import glob
import shutil
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torchvision.transforms.functional as TF
    from utils.metrics import psnr, ssim
    from models.dvc_model import DVCModel
except ImportError as e:
    print(f"Import error ({e}). Please run from the project root with deps installed.")
    sys.exit(1)


def run_cmd(cmd):
    subprocess.run(cmd, shell=True, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_bpp(file_path, num_frames, h, w):
    size_bytes = os.path.getsize(file_path)
    total_pixels = num_frames * h * w
    return (size_bytes * 8) / total_pixels


def stage_frames(frame_files, stage_dir):
    """Copy frames to 1-based contiguous names (0001.png...) so ffmpeg's %d
    pattern and the decoded output line up regardless of original naming."""
    if os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)
    for i, src in enumerate(frame_files, start=1):
        shutil.copy2(src, os.path.join(stage_dir, f"{i:04d}.png"))
    return os.path.join(stage_dir, "%04d.png")


def calculate_metrics(stage_dir, dec_prefix, num_frames):
    """Average PSNR/SSIM of H.264-decoded frames vs staged originals (1-based)."""
    psnr_list, ssim_list = [], []
    for i in range(1, num_frames + 1):
        orig_path = os.path.join(stage_dir, f"{i:04d}.png")
        dec_path = f"{dec_prefix}_{i}.png"
        if not os.path.exists(orig_path) or not os.path.exists(dec_path):
            continue
        img_orig = TF.to_tensor(Image.open(orig_path).convert("RGB")).unsqueeze(0)
        img_dec = TF.to_tensor(Image.open(dec_path).convert("RGB")).unsqueeze(0)
        psnr_list.append(psnr(img_dec, img_orig))
        ssim_list.append(ssim(img_dec, img_orig))
    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def evaluate_h264(frame_files, fps, output_dir, num_frames, h, w,
                  crfs=(18, 23, 28, 33, 38)):
    """Encode the sequence with libx264 at several CRFs -> real R-D points."""
    print("--- Evaluating H.264 baseline (libx264) ---")
    stage_dir = os.path.join(output_dir, "_src")
    input_pattern = stage_frames(frame_files, stage_dir)
    bpps, psnrs, ssims = [], [], []

    for crf in crfs:
        mp4_path = os.path.join(output_dir, f"h264_crf{crf}.mp4")
        dec_prefix = os.path.join(output_dir, f"h264_crf{crf}_dec")

        enc_cmd = (f'ffmpeg -y -framerate {fps} -i "{input_pattern}" '
                   f'-c:v libx264 -crf {crf} -preset medium -pix_fmt yuv420p '
                   f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" "{mp4_path}"')
        run_cmd(enc_cmd)
        run_cmd(f'ffmpeg -y -i "{mp4_path}" "{dec_prefix}_%d.png"')

        bpp = get_bpp(mp4_path, num_frames, h, w)
        avg_psnr, avg_ssim = calculate_metrics(stage_dir, dec_prefix, num_frames)
        bpps.append(bpp)
        psnrs.append(avg_psnr)
        ssims.append(avg_ssim)
        print(f"  CRF={crf:2d} | bpp={bpp:.4f} | PSNR={avg_psnr:.2f}dB | SSIM={avg_ssim:.4f}")

        for f in glob.glob(f"{dec_prefix}_*.png"):
            os.remove(f)

    return np.array(bpps), np.array(psnrs), np.array(ssims)


def load_dvc(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('args', {})
    model = DVCModel(flow_M=cfg.get('flow_M', 128),
                     res_M=cfg.get('res_M', 128),
                     lmbda=cfg.get('lmbda', 512))
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def evaluate_dvc(model, frame_paths, device):
    """Run the learned codec over the sequence, propagating the RECONSTRUCTED
    frame as the next reference (honest decoder behaviour). Returns mean R-D."""
    def load(p):
        return TF.to_tensor(Image.open(p).convert("RGB")).unsqueeze(0).to(device)

    ref = load(frame_paths[0])
    bpps, psnrs, ssims = [], [], []
    for p in frame_paths[1:]:
        cur = load(p)
        frame_rec, losses = model(ref, cur)
        bpps.append(losses['bpp'].item())
        psnrs.append(psnr(cur, frame_rec))
        ssims.append(ssim(cur, frame_rec))
        ref = frame_rec.detach()
    return float(np.mean(bpps)), float(np.mean(psnrs)), float(np.mean(ssims))


def main():
    parser = argparse.ArgumentParser(description="Honest R-D comparison: learned DVC vs H.264")
    parser.add_argument("--input_dir", type=str, default="data/real_test/cur",
                        help="Directory of consecutive PNG frames (any naming)")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--output_dir", type=str, default="results/h264_eval")
    parser.add_argument("--checkpoints", type=str, nargs="*", default=None,
                        help="Trained DVC checkpoints (one per lambda). Real R-D points.")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    num_frames = len(frame_files)
    if num_frames == 0:
        print(f"Error: no PNG frames in {args.input_dir}")
        return
    w, h = Image.open(frame_files[0]).size
    print(f"Found {num_frames} frames of size {w}x{h} in {args.input_dir}\n")

    h264_bpps, h264_psnrs, h264_ssims = evaluate_h264(
        frame_files, args.fps, args.output_dir, num_frames, h, w)

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))

    dvc_points = []  # (bpp, psnr, label)
    if args.checkpoints:
        print("\n--- Evaluating learned DVC model(s) ---")
        for ckpt in args.checkpoints:
            if not os.path.exists(ckpt):
                print(f"  [skip] not found: {ckpt}")
                continue
            model, cfg = load_dvc(ckpt, device)
            bpp, ps, ss = evaluate_dvc(model, frame_files, device)
            label = f"λ={cfg.get('lmbda', '?')}"
            dvc_points.append((bpp, ps, label))
            print(f"  {label:10s} | bpp={bpp:.4f} | PSNR={ps:.2f}dB | SSIM={ss:.4f}")
    else:
        print("\n[!] No --checkpoints given: plotting H.264 baseline only "
              "(no fabricated DVC numbers).")

    # ---- Plot (no extrapolation, only measured points) ----
    plt.figure(figsize=(10, 6))
    order = np.argsort(h264_bpps)
    plt.plot(h264_bpps[order], h264_psnrs[order], 'o-', color='black',
             label='H.264 (libx264, measured)')

    if dvc_points:
        dvc_points.sort(key=lambda t: t[0])
        bpps = [p[0] for p in dvc_points]
        psnrs = [p[1] for p in dvc_points]
        plt.plot(bpps, psnrs, 's-', color='red', label='Learned DVC (measured)')
        for bpp, ps, lbl in dvc_points:
            plt.annotate(lbl, (bpp, ps), textcoords="offset points", xytext=(6, 4), fontsize=8)

    plt.xlabel('Bit-rate (bpp)')
    plt.ylabel('PSNR (dB)')
    plt.title('Rate-Distortion: Learned DVC vs H.264 (same test sequence)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot_path = os.path.join('results', 'h264_vs_dvc_rd_curve.png')
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    print(f"\nR-D curve saved to: {plot_path}")


if __name__ == "__main__":
    main()
