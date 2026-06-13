import os
import tempfile
import subprocess

import cv2
import numpy as np
import pandas as pd
import torch
import streamlit as st

from models.dvc_model import DVCModel
from utils.metrics import psnr, ssim, ms_ssim_numpy, bitstream_bpp
from utils.visualization import flow_to_color, tensor_to_uint8

PROC_H, PROC_W = 256, 448
RAW_BPP = 24.0

st.set_page_config(page_title="Live Demo — Learned Motion Compensation", layout="wide")


@st.cache_resource
def load_model(lmbda):
    model = DVCModel(lmbda=lmbda)
    for p in (f"checkpoints/psnr_l{lmbda}/best.pth",
              f"checkpoints/psnr_l{lmbda}/latest.pth"):
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location="cpu")["model"], strict=True)
            model.eval()
            model.update(force=True)   # dựng bảng CDF cho arithmetic coding thật
            return model, p
    return None, None


def bgr_to_tensor(bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (PROC_W, PROC_H)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def heatmap_shared(r_hw, vmax):
    r = np.clip(r_hw / (vmax + 1e-8) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(r, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def zoom_crop(rgb, cx_pct, cy_pct, size=80, scale=4):
    h, w = rgb.shape[:2]
    x0 = max(0, min(w - size, int(cx_pct / 100 * w) - size // 2))
    y0 = max(0, min(h - size, int(cy_pct / 100 * h) - size // 2))
    crop = rgb[y0:y0 + size, x0:x0 + size]
    return cv2.resize(crop, (size * scale, size * scale), interpolation=cv2.INTER_NEAREST)


def write_mp4(frames_rgb, path, fps=10):
    td = tempfile.mkdtemp()
    for j, rgb in enumerate(frames_rgb):
        cv2.imwrite(os.path.join(td, f"f_{j:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(td, "f_%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", path], capture_output=True)
    return path


@torch.no_grad()
def run_sequence(model, frames_bgr, gop, progress=None):
    results = []
    ref_mc = ref_no = None
    for i, bgr in enumerate(frames_bgr):
        cur_t = bgr_to_tensor(bgr)
        cur_np = tensor_to_uint8(cur_t)
        if i % gop == 0:
            results.append({"idx": i, "type": "I", "cur": cur_np,
                            "rec": cur_np, "rec_nomc": cur_np})
            ref_mc = ref_no = cur_t
        else:
            # Có MC: nén THẬT qua model.encode_frame (arithmetic coding flow + residual).
            bs, rec_t, mid = model.encode_frame(ref_mc, cur_t, return_intermediates=True)
            flow_hat, pred_t, resid_mc = mid["flow_hat"], mid["pred"], mid["residual"]
            R_motion = bitstream_bpp(bs["motion_strings"], PROC_H, PROC_W)
            R_residual = bitstream_bpp(bs["residual_strings"], PROC_H, PROC_W)

            # No-MC: dự đoán = copy frame trước, cùng residual codec, cũng nén thật.
            resid_no = cur_t - ref_no
            n_str, n_shape = model.residual_coder.compress(resid_no)
            resid_no_hat = model.residual_coder.decompress(n_str, n_shape, (PROC_H, PROC_W))
            rec_no_t = (ref_no + resid_no_hat).clamp(0, 1)
            cur_np = tensor_to_uint8(cur_t)
            rec_np = tensor_to_uint8(rec_t)
            r_mc = resid_mc[0].abs().mean(0).cpu().numpy()
            r_no = resid_no[0].abs().mean(0).cpu().numpy()
            vmax = max(float(r_no.max()), float(r_mc.max()))
            results.append({
                "idx": i, "type": "P",
                "cur": cur_np, "rec": rec_np, "rec_nomc": tensor_to_uint8(rec_no_t),
                "flow": flow_to_color(flow_hat),
                "pred_mc": tensor_to_uint8(pred_t),
                "pred_no": tensor_to_uint8(ref_no),
                "resid_mc": heatmap_shared(r_mc, vmax),
                "resid_no": heatmap_shared(r_no, vmax),
                "psnr": psnr(cur_t, rec_t),
                "psnr_nomc": psnr(cur_t, rec_no_t),
                "ssim": ssim(cur_t, rec_t),
                "ms_ssim": ms_ssim_numpy(cur_np.astype(np.float32) / 255,
                                         rec_np.astype(np.float32) / 255),
                "bpp": R_motion + R_residual,
                "bpp_nomc": bitstream_bpp(n_str, PROC_H, PROC_W),
                "r_motion": R_motion,
                "r_residual": R_residual,
                "psnr_pred_mc": psnr(cur_t, pred_t),
                "psnr_pred_no": psnr(cur_t, ref_no),
            })
            ref_mc, ref_no = rec_t, rec_no_t
        if progress is not None:
            progress.progress((i + 1) / len(frames_bgr))
    return results


def encode_h264_sequence(frames_rgb, crf, fps=10):
    td = tempfile.mkdtemp(); src = os.path.join(td, "s"); os.makedirs(src, exist_ok=True)
    for j, rgb in enumerate(frames_rgb, 1):
        cv2.imwrite(os.path.join(src, f"{j:04d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    mp4 = os.path.join(td, "h.mp4")
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(src, "%04d.png"),
                    "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                    "-pix_fmt", "yuv420p", mp4], capture_output=True)
    dec = os.path.join(td, "d"); os.makedirs(dec, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", mp4, os.path.join(dec, "%04d.png")], capture_output=True)
    n = len(frames_rgb)
    bpp = (os.path.getsize(mp4) * 8) / (n * PROC_H * PROC_W) if os.path.exists(mp4) else 0.0
    recs, ps, ms = [], [], []
    for j in range(1, n + 1):
        p = os.path.join(dec, f"{j:04d}.png")
        rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) if os.path.exists(p) else frames_rgb[j - 1]
        recs.append(rgb)
        o, rr = frames_rgb[j - 1].astype(np.float32) / 255, rgb.astype(np.float32) / 255
        ps.append(psnr(o, rr)); ms.append(ms_ssim_numpy(o, rr))
    return recs, bpp, ps, ms


def match_h264_bitrate(frames_rgb, target_bpp, fps=10):
    lo, hi, best = 18, 45, None
    for _ in range(6):
        mid = (lo + hi) // 2
        recs, bpp, ps, ms = encode_h264_sequence(frames_rgb, mid, fps)
        if best is None or abs(bpp - target_bpp) < abs(best[1] - target_bpp):
            best = (recs, bpp, ps, ms, mid)
        if bpp > target_bpp:
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break
    return best


st.title("Live Demo — Learned Motion Compensation (PWCNet + DVC)")

st.sidebar.header("Cấu hình")
lambda_choice = st.sidebar.select_slider("Model (λ)", options=[256, 512, 1024], value=512)
up = st.sidebar.file_uploader("Import video", type=["mp4", "avi", "mov"])
num_frames = st.sidebar.slider("Số frame xử lý", 2, 100, 16)
gop = st.sidebar.slider("GOP (khoảng cách I-frame)", 2, 20, 8)

if st.sidebar.button("Run", type="primary"):
    if up is None:
        st.sidebar.error("Hãy tải lên một video.")
    else:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tf.write(up.read())
        video_path = tf.name
        with st.spinner(f"Tải model (λ={lambda_choice})…"):
            model, _ = load_model(lambda_choice)
        if model is None:
            st.error(f"Không có checkpoint cho λ={lambda_choice}.")
            st.stop()
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        src_fps = src_fps if src_fps and src_fps > 0 else 25.0  # giữ fps gốc cho phát lại
        frames = []
        while len(frames) < num_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if len(frames) < 2:
            st.error("Không đọc đủ frame.")
            st.stop()
        prog = st.progress(0.0)
        results = run_sequence(model, frames, gop, prog)
        prog.empty()
        td = tempfile.mkdtemp()
        videos = {
            "orig": write_mp4([x["cur"] for x in results], os.path.join(td, "orig.mp4"), src_fps),
            "nomc": write_mp4([x["rec_nomc"] for x in results], os.path.join(td, "nomc.mp4"), src_fps),
            "mc":   write_mp4([x["rec"] for x in results], os.path.join(td, "mc.mp4"), src_fps),
        }
        dvc_bpp = float(np.mean([x["bpp"] for x in results if x["type"] == "P"]))
        with st.spinner("Dò CRF để H.264 cùng bitrate…"):
            h264 = match_h264_bitrate([x["cur"] for x in results], dvc_bpp)
        videos["h264"] = write_mp4(h264[0], os.path.join(td, "h264.mp4"), src_fps)
        st.session_state.update(results=results, lmbda=lambda_choice,
                                h264=h264, dvc_bpp=dvc_bpp, videos=videos)


if "results" not in st.session_state:
    st.info("Tải lên video ở thanh bên rồi bấm 🚀 Chạy.")
    st.stop()

results = st.session_state["results"]
p_frames = [r for r in results if r["type"] == "P"]
if not p_frames or "rec_nomc" not in p_frames[0]:
    st.warning("Kết quả từ phiên chạy cũ — bấm 🚀 Chạy lại.")
    st.stop()

p_indices = [r["idx"] for r in p_frames]
sel = st.select_slider("Chọn P-frame", options=p_indices, value=p_indices[0])
r = next(x for x in results if x["idx"] == sel)

st.markdown(f"### Phân tích frame {sel}")
c1, c2, c3 = st.columns(3)
c1.image(r["cur"],     caption="Ground truth Iₜ (frame thật)")
c2.image(r["pred_mc"], caption=f"Prediction MC (warp Iₜ₋₁)")
c3.image(r["flow"],    caption="Motion field (optical flow)")
c4, c5, c6 = st.columns(3)
c4.image(r["resid_no"], caption="Residual No-MC (cur − Iₜ₋₁)")
c5.image(r["resid_mc"], caption="Residual MC ")
c6.markdown(
    f"| Metrics | Value|\n|---|---|\n"
    f"| PSNR tái tạo | **{r['psnr']:.2f} dB** |\n"
    f"| bpp | **{r['bpp']:.4f}** |\n"
    f"| R_motion | **{r['r_motion']:.4f}** |\n"
    f"| R_residual | **{r['r_residual']:.4f}** |\n"
    f"| Tỉ số nén | **{RAW_BPP / r['bpp']:.0f}×** |\n"
)

def _avg(k):
    return float(np.mean([x[k] for x in p_frames]))

st.markdown("#### Trung bình (toàn bộ P-frames)")
avg_row = pd.DataFrame({
    "PSNR tái tạo (dB)": [_avg("psnr")],
    "SSIM":             [_avg("ssim")],
    "MS-SSIM":          [_avg("ms_ssim")],
    "bpp":              [_avg("bpp")],
    "Tỉ số nén (×)":    [RAW_BPP / _avg("bpp")],
    "R_motion (bpp)":   [_avg("r_motion")],
    "R_residual (bpp)": [_avg("r_residual")],
}, index=["Trung bình"]).round(4)
st.table(avg_row)

st.markdown("### Compare with no Motion compensation")
vids = st.session_state["videos"]
bpp_no = np.mean([x["bpp_nomc"] for x in p_frames])
psnr_no = np.mean([x["psnr_nomc"] for x in p_frames])
psnr_mc = np.mean([x["psnr"] for x in p_frames])
v1, v2, v3 = st.columns(3)
v1.markdown("**Gốc**"); v1.video(vids["orig"])
v2.markdown(f"**no MC** · bpp {bpp_no:.4f} · psnr {psnr_no:.1f} dB"); v2.video(vids["nomc"])
v3.markdown(f"**MC** · bpp {st.session_state['dvc_bpp']:.4f} · psnr {psnr_mc:.1f} dB"); v3.video(vids["mc"])


recs_h, bpp_h, ps_h, ms_h, crf_h = st.session_state["h264"]
st.markdown("### Compare with H.264 (cùng bitrate)")
b1, b2 = st.columns(2)
b1.metric(f"H.264 · bpp {bpp_h:.4f}",
          f"PSNR {np.mean(ps_h):.2f} dB", f"MS-SSIM {np.mean(ms_h):.4f}", delta_color="off")
b2.metric(f"Codec học (λ={st.session_state['lmbda']}) · bpp {st.session_state['dvc_bpp']:.4f}",
          f"PSNR {psnr_mc:.2f} dB",
          f"MS-SSIM {np.mean([x['ms_ssim'] for x in p_frames]):.4f}", delta_color="off")

hv1, hv2, hv3 = st.columns(3)
hv1.markdown("**Gốc**"); hv1.video(vids["orig"])
hv2.markdown(f"**H.264** · bpp {bpp_h:.4f} · psnr {np.mean(ps_h):.1f} dB"); hv2.video(vids["h264"])
hv3.markdown(f"**Learned** · bpp {st.session_state['dvc_bpp']:.4f} · psnr {psnr_mc:.1f} dB"); hv3.video(vids["mc"])

st.markdown("**Soi từng frame**")
a1, a2, a3 = st.columns(3)
a1.image(r["cur"],    caption="Gốc")
a2.image(recs_h[sel], caption=f"H.264 · {ps_h[sel]:.2f} dB")
a3.image(r["rec"],    caption=f"Codec học · {r['psnr']:.2f} dB")

st.markdown("**🔍 Zoom**")
zx, zy = st.columns(2)
cx = zx.slider("Vị trí ngang (%)", 0, 100, 50, key="zx")
cy = zy.slider("Vị trí dọc (%)", 0, 100, 50, key="zy")
z1, z2, z3 = st.columns(3)
z1.image(zoom_crop(r["cur"], cx, cy),    caption="Gốc")
z2.image(zoom_crop(recs_h[sel], cx, cy), caption="H.264 — chú ý ô vuông")
z3.image(zoom_crop(r["rec"], cx, cy),    caption="Codec học")

