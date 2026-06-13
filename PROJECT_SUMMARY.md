# Project Summary — Learned Motion Compensation (Học máy bù chuyển động cho nén video)

> Tài liệu này tổng hợp TOÀN BỘ kiến trúc, pipeline, vai trò file và cách hoạt động
> của dự án, đủ để một AI/người khác hiểu hệ thống mà chưa cần đọc code.
> Môn học: **Nén & Mã hóa Dữ liệu Đa phương tiện**. Đề tài: **Learned Motion Compensation**.

---

## 1. Mục tiêu & phạm vi

- **Đề bài:** Neural optical flow model → xuất **Motion Fields** + **Residuals** → demo **Visual Artifact Reduction**, kèm **live demo có tương tác**.
- **Phương pháp tham khảo:**
  - DVC — Lu et al., *"DVC: An End-to-End Deep Video Compression Framework"*, CVPR 2019.
  - Scale Hyperprior — Ballé et al., *"Variational Image Compression with a Scale Hyperprior"*, ICLR 2018 (qua thư viện **CompressAI**).
- **Loại codec:** nén **liên-frame (inter-frame)** kiểu P-frame: dự đoán frame hiện tại từ frame trước bằng optical flow, chỉ mã hóa phần sai (residual).
- **Định vị trung thực:** đây là bản DVC thu nhỏ, train trên ít dữ liệu — **không nhằm đánh bại H.264**, mà để minh họa pipeline learned motion compensation và sự khác biệt loại artifact.

---

## 2. Pipeline tổng thể

Một video được tách frame, resize về **256×448**, chia thành **GOP** (Group of Pictures):

- **I-frame** (frame đầu mỗi GOP): giữ nguyên, làm reference (keyframe).
- **P-frame** (các frame còn lại): nén theo các bước:

```
1. flow      = PWCNet(ref, cur)            # ước lượng optical flow (dense, 1 vector/pixel)
2. flow_hat  = nén+giải mã flow            # MotionCompressor (VAE scale-hyperprior)
3. pred      = warp(ref, flow_hat)         # motion compensation (bilinear warping)
4. residual  = cur - pred                  # phần warp không dựng được
5. res_hat   = nén+giải mã residual        # ResidualCompressor (VAE scale-hyperprior)
6. rec       = clamp(pred + res_hat, 0,1)  # frame tái tạo
7. ref ← rec                               # propagate (gây drift; reset ở I-frame)

bpp = (số byte của flow + số byte của residual) / số pixel
```

**Hai chế độ chạy cùng một mạng (điểm cần nhấn mạnh):**

| | Train (`DVCModel.forward`) | Nén/Demo (`DVCModel.encode_frame`) |
|---|---|---|
| Lượng tử hóa | cộng nhiễu đều → **khả vi** | làm tròn + **arithmetic coding thật** |
| Rate (bpp) | ước lượng `−log₂ P(y)` (khả vi) | **đếm byte thật** của bitstream |
| Mục đích | tính loss → backprop → cập nhật trọng số | tạo bitstream + frame tái tạo thật |
| Cần `update()`? | Không | Có (dựng bảng CDF) |

→ Hai chế độ **nối nhau qua checkpoint (bộ trọng số đã học)**. Train tạo ra trọng số; nén dùng đúng trọng số đó để viết byte. Vì cùng một entropy model `P(y)` nên **bpp ước lượng ≈ bpp thật** (~1%).

---

## 3. Kiến trúc chi tiết

### 3.1 Neural optical flow — PWCNet (`models/flow_net.py`)
Mạng PWC-Net tự viết, **train from scratch** (không pretrained).
- **FeatureExtractor:** kim tự tháp 4 tầng, mỗi tầng `Conv + BatchNorm + LeakyReLU(0.1)`, stride 2. Kênh: 3 → 16 (/2) → 32 (/4) → 64 (/8) → 96 (/16).
- **CostVolume** (`max_disp=4`): tương quan đặc trưng ref↔cur trong cửa sổ ±4 → `(2·4+1)² = 81` kênh.
- **Coarse-to-fine:** decoder ở các scale /16, /8, /4; mỗi `FlowDecoder` là chuỗi conv 128→96→64→32 → 2 kênh flow (lớp cuối **init = 0**). Giữa các tầng: upsample flow ×2, warp đặc trưng, cộng dồn residual flow.
- **ContextNet:** dilated conv (hệ số 1,2,4,8,16,1) tinh chỉnh flow ở /4.
- **Output:** interpolate flow ×4 về full-res (2×H×W).
- **`bilinear_warp(frame, flow)`:** `grid_sample`, `padding_mode='border'`, `align_corners=True`.
- **Hạn chế:** `max_disp=4` + train trên frame liên tiếp → chỉ xử lý tốt **chuyển động nhỏ**.

### 3.2 Entropy coding — Scale Hyperprior VAE (`models/entropy_coder.py`)
Class `HyperpriorCompressor` (dùng CompressAI):
- **g_a (analysis):** 4× `Conv(k5,s2)` + **GDN** → latent `y`, **downsample /16**, M kênh.
- **g_s (synthesis):** 4× `ConvTranspose(k5,s2)` + **IGDN** → dựng lại tín hiệu.
- **h_a (hyper-encoder):** `|y|` → `z` (side-info, nhỏ hơn nữa).
- **h_s (hyper-decoder):** `z_hat` → **scales σ** cho từng phần tử của `y`.
- **EntropyBottleneck(N):** mã hóa `z` bằng prior factorized.
- **GaussianConditional:** mã hóa `y` ~ `N(0, σ)` với σ do h_s dự đoán.
- Tham số mặc định: `N=128, M=128`.
- **MotionCompressor:** `in_ch=2` (flow x,y). **ResidualCompressor:** `in_ch=3` (residual RGB).
- `compress()/decompress()`: arithmetic coding thật. `update()`: dựng bảng CDF (scale table hình học 0.11 → 256, 64 mức).
- `rate_estimate(likelihoods, num_pixels)`: `bits = Σ −log₂(P)`, `bpp = bits / pixel`.

### 3.3 Lắp ghép — DVCModel (`models/dvc_model.py`)
- `__init__(flow_M=128, res_M=128, lmbda=512)`: gồm `flow_net`, `motion_coder`, `residual_coder`.
- `forward(ref, cur)` → `(rec, losses)` với `losses = {loss_psnr, loss_msssim, R_motion, R_residual, R_total, D_mse, D_msssim, bpp}`. (Bản khả vi, cho train.)
- `encode_frame(ref, cur, return_intermediates=False)` → `(bitstream, rec[, {flow_hat, pred, residual}])`. (Bản nén thật — **nơi DUY NHẤT chứa logic codec thật**, dùng chung cho cả demo.)
- `update(force=True)`: dựng bảng CDF trước khi nén thật.
- `ms_ssim(...)` + `_gaussian_window(...)`: MS-SSIM 3 mức (torch) cho `loss_msssim`.

---

## 4. Loss & huấn luyện (`train.py`)

- **Hàm loss (Rate-Distortion):** `L = λ · D + R_total`
  - `D = MSE` (chế độ `--metric psnr`) hoặc `D = 1 − MS-SSIM` (`--metric msssim`).
  - `R_total = R_motion + R_residual` (ước lượng entropy).
  - **λ** cân bằng chất lượng vs bitrate. λ lớn → ưu tiên chất lượng (bpp cao).
- **λ khuyến nghị:** PSNR mode: 256/512/1024/2048 · MS-SSIM mode: 8/16/32/64.
- **Hai optimizer:**
  - chính: Adam `lr=1e-4` cho mọi tham số trừ `.quantiles`.
  - phụ (aux): Adam `lr=1e-3` cho `EntropyBottleneck.quantiles` (tối thiểu `aux_loss`).
- Gradient clipping `1.0`; `MultiStepLR` giảm lr ×0.1 tại 80% và 90% số epoch.
- Lưu `best.pth` (theo val PSNR), `latest.pth`, `config.json`, `history.json`.
- **Checkpoint hiện có:** `checkpoints/psnr_l256`, `psnr_l512`, `psnr_l1024` (đều train ở 256×448).

---

## 5. Dữ liệu

- **`extract_custom_videos.py`:** đọc `data/raw_videos/*.mp4`, tách thành **cặp (ref, cur)** = frame liên tiếp, lưu PNG ở `data/real/{ref,cur}/`; **giữ riêng 1 clip cho test** (`data/real_test/`). Tham số: `--height 256 --width 448 --stride 1 --max_pairs`.
- **`data/dataset.py`:**
  - `FramePairDataset`: nạp cặp PNG, resize, augment (lật ngang/dọc, đổi sáng).
  - `get_dataloaders`: chia train/val 90/10 (`random_split`, seed 42).

---

## 6. Cấu trúc file & vai trò

```
app.py                    # LIVE DEMO (Streamlit) — màn trình diễn duy nhất
train.py                  # Vòng huấn luyện Rate-Distortion
train_all.sh              # Script train nhiều λ
extract_custom_videos.py  # Chuẩn bị dữ liệu: .mp4 -> cặp (ref,cur) PNG
data/dataset.py           # Dataloader cặp frame
models/
  flow_net.py             # PWCNet (optical flow) + bilinear_warp
  entropy_coder.py        # 2 VAE scale-hyperprior (flow, residual) + rate_estimate
  dvc_model.py            # Ghép pipeline; forward (train) + encode_frame (nén thật)
utils/
  metrics.py              # psnr, ssim, ms_ssim_numpy, bitstream_bpp
  visualization.py        # tensor_to_uint8, flow_to_color
checkpoints/              # psnr_l256 / psnr_l512 / psnr_l1024 (best.pth, latest.pth, config.json)
data/raw_videos/          # video gốc để demo/chuẩn bị dữ liệu
```

---

## 7. Live demo (`app.py`)

Streamlit. Luồng:
1. **Sidebar:** chọn `λ` (256/512/1024), **upload video**, số frame, GOP. Nút **Run**.
2. **`load_model(λ)`** (cache): dựng `DVCModel`, nạp `checkpoints/psnr_l{λ}/best.pth`, gọi `update()` (dựng CDF cho nén thật).
3. Đọc video (giữ **fps gốc** để phát lại đúng tốc độ).
4. **`run_sequence`** (lõi): lặp từng frame, GOP-aware:
   - I-frame: reset reference.
   - P-frame: gọi `model.encode_frame(ref, cur, return_intermediates=True)` → bitstream + rec + {flow_hat, pred, residual}; tính `R_motion/R_residual` bằng `bitstream_bpp` (byte thật). Đồng thời tính **nhánh No-MC** (dự đoán = copy frame trước, cùng `residual_coder`) để so sánh.
5. Xuất **4 video mp4** (Gốc, No-MC, Có-MC, H.264) ở fps gốc; chạy `match_h264_bitrate` (binary-search CRF để H.264 **cùng bitrate** → so công bằng).
6. **Hiển thị:**
   - **Lưới phân tích frame:** Ground truth · Prediction MC · Motion field · Residual No-MC · Residual MC · **bảng chỉ số** (PSNR/bpp/R_motion/R_residual/tỉ số nén).
   - **Bảng trung bình** toàn bộ P-frames.
   - **Compare with No Motion Compensation:** 3 video (Gốc / No-MC / MC).
   - **Compare with H.264:** thẻ metric + 3 video + ảnh 3-way + **zoom** (lộ blocking).

Hằng số: `PROC_H, PROC_W = 256, 448` (chia hết 16 cho downsample /16); `RAW_BPP = 24` (RGB 8-bit) để tính tỉ số nén.

---

## 8. Kết quả đo được (thật, checkpoint psnr_l512, 256×448)

- **1 P-frame:** ~**32.7 dB @ 0.085 bpp** (bpp thật ≈ ước lượng, lệch ~1%).
- **Drift qua GOP:** ~32.7 dB (đầu) tụt dần ~29 dB (cuối GOP ~12 frame), bpp tăng nhẹ.
- **Ablation No-MC vs MC (chứng minh giá trị motion compensation):**
  - PSNR dự đoán: copy-frame ~23.8 dB → warp ~24.8 dB (**+1 dB**).
  - Năng lượng residual giảm ~**11%**.
- **So H.264 (cùng bitrate):** H.264 thường **nhỉnh hơn cả PSNR lẫn MS-SSIM** — codec học nhỏ khó vượt chuẩn đã tối ưu.
- **Blockiness:** H.264 ≈ 0.244 vs codec học ≈ 0.104 (**ít vỡ block hơn ~2.3×**) — đây là điểm "visual artifact reduction".

---

## 9. Hạn chế đã biết (trình bày trung thực)

1. **Model nhỏ + ít dữ liệu** → không thắng H.264 về PSNR/MS-SSIM ở cùng bitrate.
2. **PWCNet `max_disp=4`** → chuyển động lớn (stride lớn) làm warp kém đi.
3. **Drift** tích lũy trong GOP (dùng frame tái tạo làm reference) — giảm bằng GOP nhỏ.
4. **MC đóng góp khiêm tốn (+1 dB)**; phần lớn chất lượng do residual codec học.
5. **No-MC dùng lại residual codec đã train cho MC** → là *cận trên* lợi ích MC (lệch phân bố train); lợi ích cơ bản được chứng minh độc lập qua prediction-PSNR & residual energy.
6. Resize cứng 256×448 → video khác tỉ lệ 16:9 sẽ hơi méo.

---

## 10. Môi trường & phụ thuộc

- Python 3.12 (venv), **PyTorch 2.x**, **torchvision**, **CompressAI 1.2.8** (entropy models, GDN, EntropyBottleneck, GaussianConditional), OpenCV, NumPy, pandas, scikit-image, **Streamlit**, **FFmpeg** (libx264, cần cài hệ thống).
- Chạy demo: `streamlit run app.py`. Train: `python train.py --lmbda 1024 --metric psnr --epochs ...`.

---

## 11. Thuật ngữ nhanh

- **P-frame / I-frame:** frame dự đoán (inter) / frame khóa (intra).
- **Optical flow (dense):** trường vector chuyển động, 1 vector/pixel.
- **Warping / Motion compensation:** kéo frame trước theo flow để dự đoán frame hiện tại.
- **Residual:** sai số `cur − pred`, phần phải mã hóa.
- **bpp:** bits per pixel (bitrate). **GOP:** khoảng cách giữa hai I-frame.
- **Scale hyperprior:** entropy model dự đoán độ lệch chuẩn từng phần tử latent để mã hóa tối ưu.
- **R-D loss:** `λ·Distortion + Rate` — cân bằng chất lượng và dung lượng.
