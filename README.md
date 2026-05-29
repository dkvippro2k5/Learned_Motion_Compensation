# Learned Motion Compensation in Video Coding

**DVC-inspired | PyTorch 2.x | CompressAI | Ubuntu**

> Project môn học: Nén và Mã hóa Dữ liệu Đa Phương tiện  
> Tham khảo: Lu et al., "DVC: An End-to-End Deep Video Compression Framework", CVPR 2019  
> OpenDVC (Yang et al., 2020) — reimplemented với stack hiện đại

---

## Cấu trúc thư mục

```
learned_mc_v2/
├── models/
│   ├── flow_net.py        # PWC-Net optical flow (PyTorch 2.x)
│   ├── entropy_coder.py   # MotionCompressor, ResidualCompressor (CompressAI)
│   └── dvc_model.py       # DVCModel — full pipeline + R-D loss
├── utils/
│   ├── metrics.py         # PSNR, SSIM, bpp, BD-Rate
│   └── visualization.py   # Flow color, heatmap, comparison grid
├── data/
│   └── dataset.py         # Synthetic / Vimeo-90K / VideoFile loaders
├── train.py               # Rate-Distortion training (nhiều lambda)
├── encode.py              # Encoder: 1 P-frame hoặc video sequence
├── evaluate.py            # Đánh giá R-D curve, PSNR/SSIM/bpp
├── demo.py                # Demo nhanh với visualization
├── generate_report.py     # Tạo báo cáo PDF
├── requirements.txt
└── README.md
```

---

## Cài đặt (Ubuntu + VS Code)

```bash
# Bước 1: Cài hệ thống
sudo apt update && sudo apt install -y python3.10 python3.10-venv python3-pip git ffmpeg

# Bước 2: Tạo virtual environment
python3.10 -m venv venv_lmc
source venv_lmc/bin/activate      # Linux/macOS
# venv_lmc\Scripts\activate       # Windows

# Bước 3: Cài thư viện
pip install -r requirements.txt

# Bước 4: Kiểm tra
python -c "import torch, compressai; print('PyTorch:', torch.__version__); print('CompressAI:', compressai.__version__)"
```

### Chọn interpreter trong VS Code

1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Chọn `venv_lmc/bin/python`

---

## Sử dụng

### 1. Training

```bash
# PSNR model — lambda=512 (trung bình, synthetic data)
python train.py --lmbda 512 --metric psnr --epochs 50 --batch_size 8

# PSNR models — tất cả 4 lambda (như OpenDVC)
python train.py --lmbda 256  --metric psnr --epochs 50
python train.py --lmbda 512  --metric psnr --epochs 50
python train.py --lmbda 1024 --metric psnr --epochs 50
python train.py --lmbda 2048 --metric psnr --epochs 50

# MS-SSIM model — fine-tune từ PSNR checkpoint (như OpenDVC MS-SSIM)
python train.py --lmbda 32 --metric msssim --epochs 20 \
                --resume checkpoints/psnr_l1024/best.pth

# Training với Vimeo-90K (cần download 82GB)
python train.py --lmbda 1024 --metric psnr --epochs 100 \
                --data_root data/vimeo90k --height 256 --width 448
```

**Checkpoint lưu tại:** `checkpoints/<metric>_l<lmbda>/best.pth`

Lambda values (bám sát OpenDVC):
| Model | Lambda values |
|-------|--------------|
| PSNR  | 256, 512, 1024, 2048 |
| MS-SSIM | 8, 16, 32, 64 |

### 2. Demo

```bash
# Demo tự động (sinh dữ liệu tổng hợp)
python demo.py --checkpoint checkpoints/psnr_l512/best.pth

# Demo từ 2 ảnh cụ thể
python demo.py --checkpoint checkpoints/psnr_l512/best.pth \
               --ref path/to/frame1.png --cur path/to/frame2.png

# Demo từ video
python demo.py --checkpoint checkpoints/psnr_l512/best.pth \
               --video path/to/video.mp4 --start_frame 20
```

### 3. Encode / Decode (như OpenDVC CLI)

```bash
# Encode một P-frame (như OpenDVC_test_P-frame.py)
python encode.py \
    --ref  BasketballPass_com/f001.png \
    --raw  BasketballPass/f002.png \
    --com  BasketballPass_com/f002.png \
    --bin  BasketballPass_bin/002.bin \
    --checkpoint checkpoints/psnr_l1024/best.pth

# Encode video sequence (như OpenDVC_test_video.py)
python encode.py \
    --path BasketballPass \
    --frame 100 --GOP 10 \
    --checkpoint checkpoints/psnr_l1024/best.pth \
    --metric psnr
```

**Output directories** (như OpenDVC):
```
BasketballPass_com_psnr_1024/   # Reconstructed frames
BasketballPass_bin_psnr_1024/   # Bitstreams (.bin)
```

### 4. Đánh giá R-D curve

```bash
# Đánh giá 1 checkpoint
python evaluate.py --checkpoints checkpoints/psnr_l512/best.pth

# Đánh giá nhiều lambda → vẽ R-D curve
python evaluate.py \
    --checkpoints checkpoints/psnr_l256/best.pth \
                  checkpoints/psnr_l512/best.pth \
                  checkpoints/psnr_l1024/best.pth \
                  checkpoints/psnr_l2048/best.pth \
    --labels "λ=256" "λ=512" "λ=1024" "λ=2048"
```

### 5. Tạo báo cáo PDF

```bash
python generate_report.py
# Output: results/BaoCao_LearnedMotionCompensation_v2.pdf
```

---

## Kiến trúc

### DVCModel pipeline

```
I_{t-1}, I_t  (B, 3, H, W)
     │
     ▼
  PWCNet  ──────────────────────────── optical flow field (B, 2, H, W)
     │
     ▼
MotionCompressor ── EntropyBottleneck → flow_hat + R_motion (bpp)
     │
     ▼
bilinear_warp(I_{t-1}, flow_hat) ─────── I_pred (B, 3, H, W)
     │
     ▼
R_t = I_t - I_pred  ─────────────────── residual (B, 3, H, W)
     │
     ▼
ResidualCompressor ─ EntropyBottleneck → R_hat + R_residual (bpp)
     │
     ▼
I_rec = I_pred + R_hat  ──────────────── reconstructed frame
     │
     ▼
Loss = λ·MSE(I_rec, I_t) + R_motion + R_residual
```

### PWC-Net (Optical Flow)

| Layer | Channels | Resolution |
|-------|----------|-----------|
| FeatureExtractor L1 | 16 | H/2 |
| FeatureExtractor L2 | 32 | H/4 |
| FeatureExtractor L3 | 64 | H/8 |
| FeatureExtractor L4 | 96 | H/16 |
| CostVolume (max_disp=4) | 81 | coarse→fine |
| FlowDecoder × 3 | — | H/8 → H |
| ContextNet (dilated) | — | H (refinement) |

Total: ~1.4M params — CPU-friendly.

---

## Kết quả

| Metric | Val (epoch 17) | Demo |
|--------|---------------|------|
| PSNR   | 17.40 dB      | 25.87 dB |
| SSIM   | 0.4431        | 0.6889   |
| bpp    | 5.4220        | 5.4505   |
| Residual Entropy | — | 3.47 bits |

> Note: Val metrics thấp vì train trên 64×64 synthetic data với M=32 (nhanh). Dùng M=128 + Vimeo-90K để đạt hiệu năng đầy đủ như OpenDVC (~30+ dB).

---

## So sánh với OpenDVC gốc

| | OpenDVC | Project này |
|--|---------|------------|
| Python | 2.7/3.6 | **3.10+** |
| DL framework | TF 1.12 | **PyTorch 2.x** |
| Entropy coding | tf-compression | **CompressAI 1.2.8** |
| I-frame codec | BPG (external) | **Learned (CompressAI)** |
| Flow model | SpyNet pretrained | **PWC-Net (trainable)** |
| Loss | MSE | **Rate-Distortion λ·D+R** |
| OS support | Ubuntu 16/18 | **Ubuntu 20/22/24** |

---

## Thư viện

- **PyTorch ≥ 2.0** — deep learning framework
- **CompressAI 1.2.8** — entropy coding (thay tf-compression + BPG)
- **OpenCV ≥ 4.8** — xử lý ảnh/video
- **scikit-image** — SSIM metric
- **ReportLab** — báo cáo PDF

---

## Tài liệu tham khảo

1. Lu et al., "DVC: An End-to-End Deep Video Compression Framework", **CVPR 2019**
2. Yang et al., "OpenDVC: An Open Source Implementation of the DVC Video Compression Method", arXiv 2020
3. Yang et al., "Hierarchical Learned Video Compression (HLVC)", **CVPR 2020**
4. Sun et al., "PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume", **CVPR 2018**
5. Ballé et al., "Variational Image Compression with a Scale Hyperprior", **ICLR 2018**
6. Begaint et al., "CompressAI: A PyTorch Library for End-to-End Compression Research", 2020