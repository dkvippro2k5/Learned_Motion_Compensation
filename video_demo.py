import os, sys, argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.dvc_model import DVCModel
from utils.visualization import tensor_to_uint8

def main():
    parser = argparse.ArgumentParser(description="Create a side-by-side video demo")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best.pth")
    parser.add_argument('--video', type=str, required=True, help="Path to input video file")
    parser.add_argument('--output', type=str, default='results/output_video.mp4')
    parser.add_argument('--frames', type=int, default=30, help="Number of frames to process")
    parser.add_argument('--start_frame', type=int, default=0, help="Start frame index")
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=448)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get('args', {})
    
    model = DVCModel(flow_M=cfg.get('flow_M', 128), res_M=cfg.get('res_M', 128), lmbda=cfg.get('lmbda', 512))
    model.load_state_dict(ckpt['model'], strict=True)
    model.to(device).eval()

    cap = cv2.VideoCapture(args.video)
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 25.0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (args.width * 2, args.height))

    ret, frame = cap.read()
    if not ret:
        print("Cannot read video.")
        return

    frame = cv2.resize(frame, (args.width, args.height))
    ref_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ref_t = torch.from_numpy(ref_np.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(device)

    disp_frame = np.concatenate([frame, frame], axis=1)
    cv2.putText(disp_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(disp_frame, "Reconstructed (DVC)", (args.width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out.write(disp_frame)

    print(f"Processing {args.frames} frames from {args.video}...")
    for _ in tqdm(range(args.frames)):
        ret, cur_frame = cap.read()
        if not ret: break
        
        cur_frame = cv2.resize(cur_frame, (args.width, args.height))
        cur_np = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
        cur_t = torch.from_numpy(cur_np.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0).to(device)

        with torch.no_grad():
            frame_rec, losses = model(ref_t, cur_t)

        bpp_val = losses['bpp'].item()
        from utils.metrics import evaluate_frame
        metrics = evaluate_frame(cur_t, frame_rec, bpp_val)
        psnr_val = metrics['psnr']
        ssim_val = metrics['ms_ssim']

        rec_bgr = cv2.cvtColor(tensor_to_uint8(frame_rec[0].cpu()), cv2.COLOR_RGB2BGR)

        disp_frame = np.concatenate([cur_frame, rec_bgr], axis=1)
        cv2.putText(disp_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(disp_frame, "Reconstructed (DVC)", (args.width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        metrics_text = f"BPP: {bpp_val:.3f} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.3f}"
        cv2.putText(disp_frame, metrics_text, (args.width + 10, args.height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        out.write(disp_frame)

        ref_t = cur_t.detach()

    cap.release()
    out.release()
    print(f"Video saved to {args.output}")

if __name__ == '__main__':
    main()
