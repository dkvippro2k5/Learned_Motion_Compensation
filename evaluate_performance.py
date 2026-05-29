import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.dvc_model import DVCModel
from utils.metrics import evaluate_frame
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import FramePairDataset, generate_synthetic_frames

def get_args():
    p = argparse.ArgumentParser(description="Evaluate Performance (bpp vs PSNR & MS-SSIM)")
    p.add_argument('--checkpoints', type=str, nargs='+', required=True,
                   help='Paths to model checkpoints to evaluate')
    p.add_argument('--labels',      type=str, nargs='+', default=None,
                   help='Labels for the models (grouping by label draws a curve)')
    p.add_argument('--data_root',   type=str, default='data/synthetic')
    p.add_argument('--height',      type=int, default=128)
    p.add_argument('--width',       type=int, default=128)
    p.add_argument('--output_dir',  type=str, default='results')
    p.add_argument('--device',      type=str, default='auto')
    return p.parse_args()

def load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get('args', {})
    model = DVCModel(
        flow_M=cfg.get('flow_M', 128),
        res_M=cfg.get('res_M',  128),
        lmbda=cfg.get('lmbda',  512),
    )
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device).eval()
    return model, cfg

@torch.no_grad()
def evaluate_model(model, loader, device):
    all_psnr, all_msssim, all_bpp = [], [], []

    for ref, cur in tqdm(loader, desc='  Đang đánh giá', leave=False):
        ref, cur = ref.to(device), cur.to(device)
        frame_rec, losses = model(ref, cur)

        for i in range(ref.size(0)):
            # Tính toán các chỉ số PSNR, MS-SSIM qua hàm evaluate_frame
            res = evaluate_frame(cur[i:i+1], frame_rec[i:i+1], bpp_val=losses['bpp'].item())
            all_psnr.append(res['psnr'])
            all_msssim.append(res['ms_ssim'])
            all_bpp.append(res['bpp'])

    return {
        'psnr':    {'mean': np.mean(all_psnr), 'std': np.std(all_psnr)},
        'ms_ssim': {'mean': np.mean(all_msssim), 'std': np.std(all_msssim)},
        'bpp':     {'mean': np.mean(all_bpp),  'std': np.std(all_bpp)},
        'n': len(all_psnr),
    }

def plot_rd_curves(rd_points, labels, save_path):
    # Tạo đồ thị với 2 hàng, 1 cột
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    colors = ['#2563EB', '#DC2626', '#059669', '#D97706', '#7C3AED']

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx, (points, label) in enumerate(zip(rd_points, labels)):
        bpp_vals  = [p['bpp']['mean'] for p in points]
        psnr_vals = [p['psnr']['mean'] for p in points]
        msssim_vals = [p['ms_ssim']['mean'] for p in points]
        c = colors[idx % len(colors)]

        # Sắp xếp các điểm theo bpp tăng dần để vẽ đường cong chuẩn xác
        sort_idx = np.argsort(bpp_vals)
        bpp_vals = np.array(bpp_vals)[sort_idx]
        psnr_vals = np.array(psnr_vals)[sort_idx]
        msssim_vals = np.array(msssim_vals)[sort_idx]

        # Vẽ đường cong
        axes[0].plot(bpp_vals, psnr_vals, 'o-', color=c, label=label, linewidth=2.5, markersize=8)
        axes[1].plot(bpp_vals, msssim_vals, 'o-', color=c, label=label, linewidth=2.5, markersize=8)

    # Trục tung (Hàng trên) - PSNR
    axes[0].set_xlabel('Trục hoành: Bits per pixel (bpp) - Dung lượng (Càng nhỏ càng tốt)', fontsize=11)
    axes[0].set_ylabel('Trục tung: PSNR (dB) - Chất lượng khách quan', fontsize=11)
    axes[0].set_title('Biểu đồ Rate-Distortion: PSNR (Cao và dịch trái là lý tưởng)', fontsize=13, fontweight='bold')
    axes[0].legend(title='Mô hình', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Trục tung (Hàng dưới) - MS-SSIM
    axes[1].set_xlabel('Trục hoành: Bits per pixel (bpp) - Dung lượng (Càng nhỏ càng tốt)', fontsize=11)
    axes[1].set_ylabel('Trục tung: MS-SSIM - Chất lượng cảm nhận thị giác', fontsize=11)
    axes[1].set_title('Biểu đồ Rate-Distortion: MS-SSIM (Gần 1 và dịch trái là lý tưởng)', fontsize=13, fontweight='bold')
    axes[1].legend(title='Mô hình', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Đánh Giá Hiệu Năng Nén Video\n(Dựa trên tiêu chí Dung Lượng vs Chất Lượng)', 
                 fontsize=16, y=0.96, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] Đã lưu đồ thị Rate-Distortion tại: {save_path}")

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
             if args.device == 'auto' else torch.device(args.device)

    # Khởi tạo data
    if not os.path.exists(os.path.join(args.data_root, 'ref')):
        print("Đang tạo dữ liệu giả lập (synthetic) để test...")
        generate_synthetic_frames(100, args.height, args.width, args.data_root)
    dataset = FramePairDataset(args.data_root, args.height, args.width)
    loader  = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    # Nếu người dùng không nhập nhãn, dùng tên thư mục chứa checkpoint làm nhãn
    labels = args.labels or [os.path.basename(os.path.dirname(c)) for c in args.checkpoints]

    all_results = []
    print(f"Bắt đầu đánh giá {len(args.checkpoints)} checkpoint(s) trên {len(dataset)} cặp khung hình...")

    for ckpt_path, label in zip(args.checkpoints, labels):
        print(f"\n[{label}] Đang xử lý: {ckpt_path}")
        model, cfg = load_model_from_ckpt(ckpt_path, device)
        results = evaluate_model(model, loader, device)
        
        all_results.append((label, results))

        print(f"  -> PSNR:    {results['psnr']['mean']:.2f} ± {results['psnr']['std']:.2f} dB")
        print(f"  -> MS-SSIM: {results['ms_ssim']['mean']:.4f} ± {results['ms_ssim']['std']:.4f}")
        print(f"  -> bpp:     {results['bpp']['mean']:.5f} ± {results['bpp']['std']:.5f}")

    # Nhóm các kết quả theo cùng một nhãn (Label) để vẽ thành 1 đường cong (Curve)
    curves = {}
    for label, res in all_results:
        if label not in curves:
            curves[label] = []
        curves[label].append(res)
    
    plot_labels = list(curves.keys())
    plot_points = list(curves.values())

    save_path = os.path.join(args.output_dir, 'performance_rd_curve.png')
    plot_rd_curves(plot_points, plot_labels, save_path)

    print(f"\n[Hoàn thành] Đánh giá xong! Hãy kiểm tra kết quả tại {save_path}")

if __name__ == '__main__':
    main()
