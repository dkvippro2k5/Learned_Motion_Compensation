# Learned Motion Compensation in Video Coding

**OpenDVC-inspired | PyTorch 2.x | CompressAI | Ubuntu**

> **Project:** Nén và Mã hóa Dữ liệu Đa Phương tiện
> **Reference:** 
> 1. Lu et al., "DVC: An End-to-End Deep Video Compression Framework", CVPR 2019
> 2. Yang et al., "OpenDVC: An Open Source Implementation of the DVC Video Compression Method", 2020

---

## 1. Code Submission (Modular Structure)

Dự án này được tổ chức theo cấu trúc module rõ ràng chuẩn MVC, tất cả các file mã nguồn đều được đặt tên chuẩn mực kèm theo giải thích chức năng chi tiết.

```text
Learned_Motion_Compensation/
├── models/
│   ├── dvc_model.py             # Core DVC architecture (R-D loss, Full Pipeline)
│   ├── flow_net.py              # PWC-Net (Optical flow) & Bilinear Warping
│   └── entropy_coder.py         # VAEs (MotionCompressor, ResidualCompressor) with CompressAI
├── utils/
│   ├── metrics.py               # Metrics: PSNR, MS-SSIM, BPP, BD-Rate
│   └── visualization.py         # Plotting: Flow color map, Residual heatmap, Grid
├── data/
│   └── dataset.py               # Dataloaders for Vimeo-90K & Synthetic data
├── train.py                     # Main R-D training loop (MultiStepLR, Gradient Clipping)
├── evaluate.py                  # Evaluation script (R-D Curves)
├── encode.py                    # Standalone bitstream encoder (.bin generator)
├── demo.py                      # CLI Demo: 6-component visual analysis grid
├── video_demo.py                # Headless CLI Demo: Side-by-side .mp4 export
├── app.py                       # Live Demo System: Interactive Streamlit Web App
├── extract_custom_videos.py     # Data prep: Extract frames from .mp4 via FFmpeg
├── compare_h264.py              # Benchmarking against H.264 standard
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

### Setup Instructions (Linux/Ubuntu Recommended)
To ensure the code **runs without errors**, follow these exact setup instructions:

```bash
# 1. Update system & install prerequisites (FFmpeg is required for video handling)
sudo apt update && sudo apt install -y python3.10 python3.10-venv python3-pip git ffmpeg

# 2. Clone the repository
git clone https://github.com/dkvippro2k5/Learned_Motion_Compensation.git
cd Learned_Motion_Compensation

# 3. Create a virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 2. Dataset Strategy

### 2.1 Production Training (Vimeo-90K Septuplet)
Mô hình được thiết kế để huấn luyện trên tập dữ liệu chuẩn **Vimeo-90K Septuplet** (~546,000 cặp frame).
- **Download:** http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip (82 GB)
- **Input Strategy:** Hệ thống tự động Random Crop ảnh về `256x256` hoặc `128x128` với Batch Size = 8 để tối ưu GPU VRAM trong quá trình Train.

### 2.2 Final Evaluation & Live Demo (Real-World Videos)
Để kiểm thử khả năng nén thực tế, dự án sử dụng các video người thật chất lượng cao (960x540).
👉 **[LINK TẢI DATASET TEST (Google Drive)](https://drive.google.com/file/d/1_m9vBStY1V1sDS1hwjG7_ecL-HgpBtwT/view?usp=sharing)** 

**Cách sử dụng:** Tải file `raw_videos.zip`, giải nén vào thư mục `data/raw_videos/` và chạy `python extract_custom_videos.py`.

---

## 3. Live Demo System (Interactive Web App)

Dự án tích hợp sẵn một ứng dụng Web giao diện trực quan bằng Streamlit, cho phép:
- Chọn file video đầu vào.
- Tùy chỉnh hệ số Lambda (Bitrate) để xem sự thay đổi chất lượng thời gian thực.
- So sánh song song Video Gốc và Video Nén (Reconstructed).
- Hiển thị trực tiếp các chỉ số BPP, PSNR, SSIM.

**Cách chạy Live Demo:**
```bash
streamlit run app.py
```

---

## 4. Labeled Outputs (Terminal Demos)

Để xuất ra các kết quả đánh nhãn (Raw, Residuals, Encoded) một cách chi tiết phục vụ báo cáo học thuật:

**Trích xuất lưới ảnh (Visual Grid 6 thành phần):**
```bash
python demo.py --checkpoint checkpoints/psnr_l1024/best.pth
```
*Kết quả `results/demo_output.png` sẽ hiển thị rõ:*
1. Reference $I_{t-1}$
2. Current $I_t$
3. Predicted $\hat{I}_t$
4. Optical Flow (Color map)
5. Residual $|R_t|$
6. Reconstructed $I_t$ (Kèm PSNR, SSIM, BPP)

**Trích xuất Video nén tự động (Headless Batch):**
```bash
python video_demo.py \
    --checkpoint checkpoints/psnr_l1024/best.pth \
    --video data/raw_videos/13910151_960_540_24fps.mp4 \
    --frames 50 \
    --output results/test_output.mp4
```

---

## 5. Training Instructions
Hệ thống sử dụng Loss Function: $\mathcal{L} = \lambda \cdot D_{MSE} + R_{total}$ và tự động lưu Checkpoint tốt nhất.

```bash
# Train với Lambda = 1024 (Ưu tiên chất lượng hình ảnh)
python train.py --lmbda 1024 --metric psnr --epochs 500 --batch_size 8
```

---

## 6. Experimental Results
Sau 500 Epochs huấn luyện, mô hình học sâu của dự án (kiến trúc PWC-Net + VAE Entropy Bottleneck) đạt hiệu năng ổn định tại mốc cấu hình $\lambda = 1024$:
- **Bitrate (BPP):** 4.806
- **PSNR:** 32.69 dB
- **SSIM:** 0.949

Hệ thống loại bỏ hoàn toàn hiện tượng vỡ ô vuông (Blocking Artifacts) vốn là nhược điểm chí mạng của H.264/AVC ở các mức dung lượng nén thấp.