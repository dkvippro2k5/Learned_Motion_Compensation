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
├── app.py                       # Live Demo DUY NHẤT: Streamlit (nén thật + pipeline + metrics)
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

### 2.1 Training data
Mặc định `train.py` huấn luyện trên các cặp frame `(ref, cur)` trích từ video thật trong
`data/real/` (sinh bằng `extract_custom_videos.py`). Đây là tập nhỏ, đủ để minh hoạ pipeline.

> **Mở rộng (tùy chọn):** để có kết quả mạnh hơn, có thể train trên **Vimeo-90K Septuplet**
> (~546k cặp, http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip, 82 GB) bằng cách
> trỏ `--data_root` tới thư mục chứa các cặp `ref/`, `cur/` tương ứng.

### 2.2 Final Evaluation & Live Demo (Real-World Videos)
Để kiểm thử khả năng nén thực tế, dự án sử dụng các video người thật chất lượng cao (960x540).
👉 **[LINK TẢI DATASET TEST (Google Drive)](https://drive.google.com/file/d/1_m9vBStY1V1sDS1hwjG7_ecL-HgpBtwT/view?usp=sharing)** 

**Cách sử dụng:** Tải file `raw_videos.zip`, giải nén vào thư mục `data/raw_videos/` và chạy `python extract_custom_videos.py`.

---

## 3. Live Demo System (Interactive Web App)

Toàn bộ phần demo (trước đây nằm ở `demo.py` và `video_demo.py`) đã được gộp vào
**một ứng dụng Streamlit duy nhất** `app.py`, cho phép:
- Chọn video (upload / mẫu), số frame, và **GOP** (khoảng cách I-frame).
- **Pipeline view 6 thành phần** cho từng frame: Reference · Current · Predicted (warp) ·
  Motion field · Residual heatmap · Reconstructed (có slider soi bất kỳ frame nào).
- **So sánh song song** Video Gốc vs Tái tạo.
- **Biểu đồ PSNR/bpp theo frame** thể hiện rõ drift trong mỗi GOP và điểm reset ở I-frame.
- Chỉ số BPP / PSNR / SSIM / MS-SSIM trung thực.

**Cách chạy Live Demo:**
```bash
streamlit run app.py
```

> **GOP & drift:** đây là một codec liên-frame, nên chất lượng (PSNR) tụt dần trong mỗi
> GOP rồi được "làm mới" tại I-frame. GOP nhỏ hơn ⇒ ít drift nhưng bitrate cao hơn.

> **Nén thật:** `app.py` thực hiện arithmetic coding thật (CompressAI) cho cả flow và
> residual nên bpp / R_motion / R_residual hiển thị là **số đo từ bitstream thật**, không
> phải ước lượng.

---

## 4. Training Instructions
Hệ thống sử dụng Loss Function: $\mathcal{L} = \lambda \cdot D_{MSE} + R_{total}$ và tự động lưu Checkpoint tốt nhất.

```bash
# Train với Lambda = 1024 (Ưu tiên chất lượng hình ảnh)
python train.py --lmbda 1024 --metric psnr --epochs 500 --batch_size 8
```

---

## 5. Experimental Results

Kết quả đo trên video thật (960×540 → xử lý ở 256×448), checkpoint `psnr_l512`, đã
**kiểm chứng bpp ước lượng khớp ~1% với bpp nén thật bằng arithmetic coding**:

| Vị trí trong GOP | PSNR (dB) | bpp | Baseline "copy frame trước" |
|---|---|---|---|
| P-frame đầu (frame 1) | **32.7** | 0.084 | 24.3 |
| Giữa GOP (frame 6)    | 30.2 | 0.121 | 24.4 |
| Cuối GOP (frame 12)   | 29.3 | 0.130 | 24.0 |

- Motion compensation cho PSNR cao hơn baseline copy-frame ~**6–8 dB** ⇒ flow + warp
  thực sự giảm residual/artifact (đúng mục tiêu đề bài).
- PSNR tụt dần trong GOP (drift tích lũy) là hành vi bình thường của codec liên-frame;
  `app.py` chèn I-frame theo GOP để làm mới chất lượng.

> So với H.264 ở bitrate thấp, mô hình tránh được blocking artifact (vỡ ô vuông) nhờ
> warp + residual liên tục thay vì chia block; xem khối so sánh trong live demo.