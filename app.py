import streamlit as st
import torch
import cv2
import numpy as np
import os
import tempfile
import subprocess
from models.dvc_model import DVCModel
from utils.metrics import evaluate_frame

st.set_page_config(page_title="Live Demo - DVC", layout="wide")

@st.cache_resource
def load_model(lmbda):
    device = 'cpu'
    model = DVCModel(lmbda=lmbda).to(device)
    ckpt_path = f'checkpoints/psnr_l{lmbda}/latest.pth'
    
    if not os.path.exists(ckpt_path):

        ckpt_path = f'checkpoints/psnr_l{lmbda}/best.pth'
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'], strict=True)
    else:
        st.error(f"No checkpoint found for Lambda={lmbda}!")
        return None

    model.motion_coder.entropy_bottleneck.update()
    model.residual_coder.entropy_bottleneck.update()
    model.eval()
    return model

st.title("🎬 Live Demo: Learned Motion Compensation (v2)")
st.markdown("This interactive system demonstrates End-to-End Deep Video Compression (DVC) in near real-time.")

st.sidebar.header("⚙️ Configuration")

st.sidebar.markdown("### Bitrate vs Quality")
lambda_choice = st.sidebar.select_slider(
    "Select Lambda ($\lambda$)",
    options=[256, 512, 1024, 2048],
    value=1024,
    help="Higher lambda = better quality (PSNR) but higher bitrate (BPP)."
)

st.sidebar.markdown("### Input Video")
input_source = st.sidebar.radio(
    "Select Video Source",
    ["Upload Video", "Demo Video 1", "Demo Video 2"]
)

video_path = None
if input_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload an MP4 file", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
elif input_source == "Demo Video 1":
    video_path = "data/raw_videos/12843543_960_540_30fps.mp4"
else:
    video_path = "data/raw_videos/13910151_960_540_24fps.mp4"

num_frames = st.sidebar.slider("Frames to Process", min_value=2, max_value=30, value=10)

if st.sidebar.button("🚀 Process Video", type="primary"):
    if not video_path or not os.path.exists(video_path):
        st.sidebar.error("Please provide a valid video.")
    else:
        with st.spinner(f"Loading Model (Lambda={lambda_choice})..."):
            model = load_model(lambda_choice)
        
        if model is not None:

            progress_bar = st.progress(0)
            status_text = st.empty()
            
            cap = cv2.VideoCapture(video_path)

            temp_dir = tempfile.mkdtemp()
            orig_dir = os.path.join(temp_dir, 'orig')
            rec_dir = os.path.join(temp_dir, 'rec')
            os.makedirs(orig_dir, exist_ok=True)
            os.makedirs(rec_dir, exist_ok=True)

            ret, prev_frame = cap.read()
            if not ret:
                st.error("Failed to read video.")
                st.stop()

            prev_frame = cv2.resize(prev_frame, (448, 256))
            cv2.imwrite(os.path.join(orig_dir, "frame_00.png"), prev_frame)
            cv2.imwrite(os.path.join(rec_dir, "frame_00.png"), prev_frame) 
            
            prev_tensor = torch.from_numpy(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
            
            metrics = {'bpp': [], 'psnr': [], 'ssim': [], 'ms_ssim': []}
            frames_processed = 1

            while frames_processed < num_frames:
                ret, cur_frame = cap.read()
                if not ret: 
                    break
                
                cur_frame = cv2.resize(cur_frame, (448, 256))
                cur_tensor = torch.from_numpy(cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
                
                status_text.text(f"Processing Frame {frames_processed+1}/{num_frames}...")
                
                with torch.no_grad():

                    frame_rec, losses = model(prev_tensor, cur_tensor)
                    bpp = losses['bpp'].item()
                    
                rec_np = (frame_rec[0].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                rec_bgr = cv2.cvtColor(rec_np, cv2.COLOR_RGB2BGR)

                cur_rgb = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                rec_rgb = cv2.cvtColor(rec_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                frame_metrics = evaluate_frame(cur_rgb, rec_rgb, bpp_val=bpp)
                
                metrics['bpp'].append(bpp)
                metrics['psnr'].append(frame_metrics['psnr'])
                metrics['ssim'].append(frame_metrics['ssim'])
                metrics['ms_ssim'].append(frame_metrics['ms_ssim'])

                cv2.imwrite(os.path.join(orig_dir, f"frame_{frames_processed:02d}.png"), cur_frame)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(rec_bgr, f"BPP: {bpp:.2f}", (10, 30), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(rec_bgr, f"PSNR: {frame_metrics['psnr']:.2f}dB", (10, 60), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(rec_dir, f"frame_{frames_processed:02d}.png"), rec_bgr)

                prev_tensor = cur_tensor 
                
                frames_processed += 1
                progress_bar.progress(frames_processed / num_frames)
            
            cap.release()

            status_text.text("Compiling MP4 videos for playback...")
            orig_mp4 = os.path.join(temp_dir, 'original.mp4')
            rec_mp4 = os.path.join(temp_dir, 'reconstructed.mp4')
            
            subprocess.run(['ffmpeg', '-y', '-framerate', '10', '-i', os.path.join(orig_dir, 'frame_%02d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', orig_mp4], capture_output=True)
            subprocess.run(['ffmpeg', '-y', '-framerate', '10', '-i', os.path.join(rec_dir, 'frame_%02d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', rec_mp4], capture_output=True)
            
            progress_bar.empty()
            status_text.empty()
            st.success("✅ Processing Complete!")

            st.markdown("### 📊 Average Performance Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Bitrate (BPP)", f"{np.mean(metrics['bpp']):.4f}")
            m2.metric("Quality (PSNR)", f"{np.mean(metrics['psnr']):.2f} dB")
            m3.metric("Structure (SSIM)", f"{np.mean(metrics['ssim']):.4f}")
            m4.metric("MS-SSIM", f"{np.mean(metrics['ms_ssim']):.4f}")

            st.markdown("### 🎥 Side-by-Side Playback")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Video**")
                st.video(orig_mp4)
                
            with col2:
                st.markdown(f"**Compressed Video (Lambda={lambda_choice})**")
                st.video(rec_mp4)
