# Learned Motion Compensation in Video Coding

**DVC-inspired | PyTorch 2.x | CompressAI | Ubuntu**

> Project: Nén và Mã hóa Dữ liệu Đa Phương tiện
> Tham khảo: Lu et al., "DVC: An End-to-End Deep Video Compression Framework", CVPR 2019

---

## 1. Code Submission (Required)

### Clean, Runnable Code & Modular Structure

Dự án này được tổ chức theo cấu trúc module rõ ràng, tất cả các file mã nguồn đều được đặt tên chuẩn mực, kèm theo comments giải thích chức năng chi tiết trong từng file.

```text
learned_mc_v2/
├── models/
│   ├── flow_net.py        # PWC-Net optical flow (PyTorch 2.x)
│   ├── entropy_coder.py   # MotionCompressor, ResidualCompressor (CompressAI)
│   └── dvc_model.py       # DVCModel — full pipeline + R-D loss
├── utils/
│   ├── metrics.py         # PSNR, SSIM, bpp, BD-Rate
│   └── visualization.py   # Flow color, heatmap, comparison grid
├── data/
│   └── dataset.py         # Synthetic / Vimeo-90K Dataset loaders
├── train.py               # Rate-Distortion training loop
├── encode.py              # Encoder (Bitstream generation)
├── video_demo.py          # Demo: Export side-by-side .mp4 comparison
├── demo.py                # Demo: Generates frame-by-frame visual grids
├── app.py                 # Live Demo: Streamlit Interactive Web Application
├── requirements.txt       # Environment dependencies
└── README.md              # Project documentation (You are reading this)
```

### Setup Instructions & Environment File (Requirements.txt)

To ensure the code **runs without errors**, follow these exact setup instructions on a Linux machine (Ubuntu recommended):

```bash
# 1. Update system & install prerequisites (FFmpeg is required for video handling)
sudo apt update && sudo apt install -y python3.10 python3.10-venv python3-pip git ffmpeg

# 2. Clone the repository
git clone <YOUR_GITHUB_REPO_LINK>
cd Learned_Motion_Compensation

# 3. Create a virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 4. Install dependencies (from Requirements.txt)
pip install -r requirements.txt
```

---

## 2. Dataset / Test Inputs (Required)

Do giới hạn dung lượng 100MB của GitHub, kho lưu trữ này không đính kèm các file dữ liệu video thô. Tuy nhiên, dự án cung cấp phương thức linh hoạt để thoả mãn yêu cầu về Test Inputs:

### Sử dụng Video Thực Tế (Raw Data)
👉 **[LINK TẢI DATASET (Google Drive): [https://drive.google.com/file/d/1_m9vBStY1V1sDS1hwjG7_ecL-HgpBtwT/view?usp=sharing]]** 👈

**Hướng dẫn cách lấy dataset (Instructions to obtain dataset):**
1. Tải file `raw_videos.zip` từ đường link Google Drive ở trên.
2. Giải nén và đặt toàn bộ các file `.mp4` vào thư mục `data/raw_videos/`.
3. Chạy lệnh trích xuất: `python extract_custom_videos.py` để tách frame.
4. Chạy lệnh Test với video thực tế:
```bash
python video_demo.py \
    --checkpoint checkpoints/psnr_l512/best.pth \
    --video data/raw_videos/8774700-sd_960_540_25fps.mp4 \
    --frames 50 \
    --output results/test_output.mp4
```

---

## 3. Labeled Outputs (Raw, Residuals, Encoded)

Để quan sát rõ ràng các thành phần mã hóa (như yêu cầu: Raw, Residuals, Encoded), thuật toán đã được lập trình để **xuất toàn bộ các thành phần** ra một lưới ảnh (Grid Image).

**Cách chạy:**
```bash
python demo.py --checkpoint checkpoints/psnr_l512/best.pth
```
**Kết quả đầu ra (`results/demo_output.png`) sẽ được Labeled (Đánh nhãn) rõ ràng:**
1. **Reference $I_{t-1}$:** Khung hình Gốc trước đó (Raw).
2. **Current $I_t$:** Khung hình Gốc hiện tại (Raw).
3. **Predicted $\hat{I}_t$:** Khung hình Dự đoán (Chỉ dùng Optical Flow bù chuyển động).
4. **Optical Flow:** Biểu đồ màu biểu diễn hướng chuyển động.
5. **Residual $|R_t|$:** Ma trận phần dư (Lỗi sai lệch giữa Raw và Predicted).
6. **Reconstructed $I_t$:** Khung hình Đã giải nén (Encoded/Decoded Result) cộng gộp từ Predicted và Residual. Kèm theo chỉ số PSNR, SSIM, bpp.

Ngoài ra, nếu chạy `python video_demo.py`, video kết quả (`results/output_video.mp4`) sẽ được dán nhãn **"Original"** (trái) và **"Reconstructed (DVC)"** (phải) để so sánh trực quan độ mượt mà.

---

## 4. Live Demo System (Interactive)

Hệ thống được tích hợp sẵn một ứng dụng Web tương tác bằng Streamlit, cho phép:
- Tải video lên hoặc chọn video mẫu.
- Kéo thanh trượt để thay đổi Bitrate (hệ số Lambda $\lambda$).
- Xem hệ thống xử lý, so sánh video gốc và video nén theo thời gian thực.
- Cập nhật chỉ số PSNR, SSIM, BPP trực tiếp trên màn hình.

**Cách chạy Live Demo:**
```bash
streamlit run app.py
```

---

## 4. Training Instructions (For Reference)

Nếu muốn train mô hình từ đầu (From scratch):

```bash
# Train trên dữ liệu tổng hợp (nhanh)
python train.py --lmbda 512 --metric psnr --epochs 50 --batch_size 8
```
Mô hình sẽ liên tục đánh giá và tự động lưu phiên bản tốt nhất tại `checkpoints/psnr_l1024/best.pth`.