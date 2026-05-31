import os
import subprocess
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    from utils.metrics import psnr, ssim
    import torch
    import torchvision.transforms.functional as TF
except ImportError:
    print("Could not import metrics from utils.metrics. Please run from project root.")
    exit(1)

def run_cmd(cmd):
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_bpp(file_path, num_frames, h, w):
    size_bytes = os.path.getsize(file_path)

    total_pixels = num_frames * h * w
    return (size_bytes * 8) / total_pixels

def calculate_metrics(orig_dir, dec_prefix, num_frames, frame_pattern):
    psnr_list = []
    ssim_list = []
    
    for i in range(1, num_frames + 1):

        orig_path = os.path.join(orig_dir, frame_pattern % i)

        dec_path = f"{dec_prefix}_{i}.png"
        
        if not os.path.exists(orig_path) or not os.path.exists(dec_path):
            continue
            
        img_orig = TF.to_tensor(Image.open(orig_path).convert("RGB")).unsqueeze(0)
        img_dec = TF.to_tensor(Image.open(dec_path).convert("RGB")).unsqueeze(0)
        
        val_psnr = psnr(img_dec, img_orig)
        val_ssim = ssim(img_dec, img_orig)
        
        psnr_list.append(val_psnr)
        ssim_list.append(val_ssim)
        
    return np.mean(psnr_list), np.mean(ssim_list)

def main():
    parser = argparse.ArgumentParser(description="Benchmark H.264 against your DVC Model")
    parser.add_argument("--input_dir", type=str, default="data/vimeo90k_subset/sequences/00001/0001", help="Directory containing PNG frames")
    parser.add_argument("--frame_pattern", type=str, default="im%d.png", help="Pattern of input frames (e.g., im%d.png or %04d.png)")
    parser.add_argument("--fps", type=int, default=25, help="Framerate for FFmpeg")
    parser.add_argument("--output_dir", type=str, default="results/h264_eval", help="Output directory for temp files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    frame_files = glob.glob(os.path.join(args.input_dir, "*.png"))
    num_frames = len(frame_files)
    if num_frames == 0:
        print(f"Error: No PNG frames found in {args.input_dir}")
        return
        
    sample_img = Image.open(frame_files[0])
    w, h = sample_img.size
    print(f"Found {num_frames} frames of size {w}x{h} in {args.input_dir}")

    crfs = [0, 2, 6, 10, 14, 18]
    h264_bpps = []
    h264_psnrs = []
    h264_ssims = []

    print("--- Evaluating H.264 (Baseline) ---")
    input_pattern = os.path.join(args.input_dir, args.frame_pattern)
    
    for crf in crfs:
        mp4_path = os.path.join(args.output_dir, f"h264_crf{crf}.mp4")
        dec_prefix = os.path.join(args.output_dir, f"h264_crf{crf}_dec")
        
        print(f"Encoding at CRF={crf}...")
        
        enc_cmd = f'ffmpeg -y -framerate {args.fps} -i "{input_pattern}" -c:v libx264 -crf {crf} -preset veryfast -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" "{mp4_path}"'
        run_cmd(enc_cmd)
        
        dec_cmd = f'ffmpeg -y -i "{mp4_path}" "{dec_prefix}_%d.png"'
        run_cmd(dec_cmd)
        
        bpp = get_bpp(mp4_path, num_frames, h, w)
        avg_psnr, avg_ssim = calculate_metrics(args.input_dir, dec_prefix, num_frames, args.frame_pattern)
        
        h264_bpps.append(bpp)
        h264_psnrs.append(avg_psnr)
        h264_ssims.append(avg_ssim)
        
        print(f"  Result: CRF={crf} | BPP={bpp:.4f} | PSNR={avg_psnr:.2f}dB | SSIM={avg_ssim:.4f}")
        
        for dec_file in glob.glob(f"{dec_prefix}_*.png"):
            os.remove(dec_file)

    dvc_bpps = np.array([7.5151, 6.9662, 6.4143, 5.8672, 5.4108, 4.9644, 4.8693, 4.8455, 4.7878, 4.8305, 4.8065])
    dvc_psnrs = np.array([35.40, 31.26, 36.41, 34.79, 29.19, 29.72, 28.52, 34.56, 30.75, 30.52, 32.69])

    sort_idx = np.argsort(dvc_bpps)
    dvc_bpps = dvc_bpps[sort_idx]
    dvc_psnrs = dvc_psnrs[sort_idx]

    h264_coeffs = np.polyfit(np.log(h264_bpps), h264_psnrs, 1)
    h264_extrap_bpps = np.linspace(np.min(h264_bpps), 7.6, 20)
    h264_extrap_psnrs = h264_coeffs[0] * np.log(h264_extrap_bpps) + h264_coeffs[1]
    
    plt.figure(figsize=(10, 6))

    plt.plot(h264_extrap_bpps, h264_extrap_psnrs, marker='', linestyle='-', color='black', label='H.264 (Extrapolated to 7.6 bpp)')

    plt.plot(h264_bpps, h264_psnrs, marker='o', linestyle='', color='black')

    plt.plot(dvc_bpps, dvc_psnrs, marker='s', linestyle='-', color='red', label='DVC Model (Epoch 300-500 Trajectory)')
    
    plt.xlabel('Bit-rate (bpp)')
    plt.ylabel('PSNR (dB)')
    plt.title('Rate-Distortion Curve: Learned DVC vs H.264 Baseline')
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join('results', 'h264_vs_dvc_rd_curve.png')
    plt.savefig(plot_path)
    print(f"Success! R-D Curve saved to: {plot_path}")

if __name__ == "__main__":
    main()
