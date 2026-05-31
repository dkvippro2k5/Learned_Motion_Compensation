"""
train.py — Rate-Distortion training for Learned Motion Compensation v2
Inspired by DVC / OpenDVC training procedure (CVPR 2019)

Usage (chạy trong VS Code terminal):

    python train.py --lmbda 512 --metric psnr --epochs 20 --batch_size 8

    python train.py --lmbda 1024 --metric psnr --epochs 50 --data_root data/vimeo90k

    python train.py --lmbda 32 --metric msssim --epochs 20 --resume checkpoints/psnr_l1024/best.pth

Lambda values (bám sát OpenDVC):
    PSNR models:   256, 512, 1024, 2048
    MS-SSIM models: 8,  16,   32,   64
"""

import os, sys, argparse, json, time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.dvc_model import DVCModel
from utils.metrics import psnr, ssim, evaluate_frame
from data.dataset import get_dataloaders

def get_args():
    p = argparse.ArgumentParser(description='DVC-inspired Rate-Distortion training')

    p.add_argument('--lmbda',      type=int,   default=512,
                   choices=[8, 16, 32, 64, 256, 512, 1024, 2048],
                   help='R-D lambda. PSNR: 256/512/1024/2048. MS-SSIM: 8/16/32/64')
    p.add_argument('--metric',     type=str,   default='psnr',
                   choices=['psnr', 'msssim'],
                   help='Optimization target (like OpenDVC PSNR vs MS-SSIM mode)')
    p.add_argument('--epochs',     type=int,   default=1000)
    p.add_argument('--batch_size', type=int,   default=8)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--aux_lr',     type=float, default=1e-3,
                   help='Learning rate for entropy bottleneck auxiliary loss')

    p.add_argument('--flow_M',  type=int, default=128,
                   help='Motion latent channels (N in OpenDVC)')
    p.add_argument('--res_M',   type=int, default=128,
                   help='Residual latent channels (M in OpenDVC)')

    p.add_argument('--data_root',  type=str, default='data/synthetic')
    p.add_argument('--height',     type=int, default=128)
    p.add_argument('--width',      type=int, default=128)
    p.add_argument('--num_pairs',  type=int, default=400)

    p.add_argument('--checkpoint', type=str, default=None,
                   help='Output dir (auto: checkpoints/<metric>_l<lmbda>/)')
    p.add_argument('--resume',     type=str, default=None)
    p.add_argument('--device',     type=str, default='auto')

    return p.parse_args()

def setup_checkpoint_dir(args):
    if args.checkpoint:
        return args.checkpoint
    return os.path.join('checkpoints', f'{args.metric}_l{args.lmbda}')

def build_model(args):
    return DVCModel(
        flow_M=args.flow_M,
        res_M=args.res_M,
        lmbda=args.lmbda,
        use_iframe_codec=False,  # I-frame codec adds weight, skip for now
    )

def train_one_epoch(model, loader, optimizer, aux_optimizer, metric, device, show_progress=True):
    model.train()
    stats = {'loss': 0, 'R_motion': 0, 'R_residual': 0, 'bpp': 0,
             'D': 0, 'psnr': 0}
    n = 0

    iterable = tqdm(loader, desc='  train', leave=False) if show_progress else loader
    for ref, cur in iterable:
        ref, cur = ref.to(device), cur.to(device)

        frame_rec, losses = model(ref, cur)

        loss = losses['loss_psnr'] if metric == 'psnr' else losses['loss_msssim']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        aux_loss = (model.motion_coder.entropy_bottleneck.loss() +
                    model.residual_coder.entropy_bottleneck.loss())
        aux_optimizer.zero_grad()
        aux_loss.backward()
        aux_optimizer.step()

        stats['loss']      += loss.item()
        stats['R_motion']  += losses['R_motion'].item()
        stats['R_residual']+= losses['R_residual'].item()
        stats['bpp']       += losses['bpp'].item()
        stats['D']         += losses['D_mse'].item()

        with torch.no_grad():
            stats['psnr'] += psnr(cur, frame_rec)
        n += 1

    return {k: v / max(n, 1) for k, v in stats.items()}

@torch.no_grad()
def validate(model, loader, metric, device, show_progress=True):
    model.eval()
    stats = {'loss': 0, 'bpp': 0, 'psnr': 0, 'ssim': 0}
    n = 0

    iterable = tqdm(loader, desc='  val  ', leave=False) if show_progress else loader
    for ref, cur in iterable:
        ref, cur = ref.to(device), cur.to(device)
        frame_rec, losses = model(ref, cur)

        loss = losses['loss_psnr'] if metric == 'psnr' else losses['loss_msssim']
        stats['loss'] += loss.item()
        stats['bpp']  += losses['bpp'].item()
        stats['psnr'] += psnr(cur, frame_rec)
        stats['ssim'] += ssim(cur, frame_rec)
        n += 1

    return {k: v / max(n, 1) for k, v in stats.items()}

def main():
    args = get_args()
    ckpt_dir = setup_checkpoint_dir(args)
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\nDevice: {device}")
    print(f"Mode:   {args.metric.upper()}, λ={args.lmbda}")

    train_loader, val_loader = get_dataloaders(
        args.data_root, args.batch_size, args.height, args.width,
        num_workers=0,
    )

    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model:  DVCModel | params: {n_params:,}")

    parameters = [p for n, p in model.named_parameters()
                  if not n.endswith('.quantiles') and p.requires_grad]
    aux_parameters = [p for n, p in model.named_parameters()
                      if n.endswith('.quantiles') and p.requires_grad]

    optimizer     = optim.Adam(parameters,     lr=args.lr)
    aux_optimizer = optim.Adam(aux_parameters, lr=args.aux_lr)

    milestones = [int(args.epochs * 0.8), int(args.epochs * 0.9)]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    start_epoch = 0
    best_val_psnr = 0.0
    history = {'train': [], 'val': []}

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        if 'optimizer' in ckpt:
            try: optimizer.load_state_dict(ckpt['optimizer'])
            except: pass
        start_epoch = ckpt.get('epoch', -1) + 1
        best_val_psnr = ckpt.get('best_val_psnr', 0.0)
        history = ckpt.get('history', history)
        print(f"Resumed from epoch {start_epoch} | best PSNR: {best_val_psnr:.2f}")

    with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nTraining {args.epochs} epochs → {ckpt_dir}/")
    print("=" * 65)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        
        is_log_epoch = ((epoch + 1) % 20 == 0) or (epoch == 0) or (epoch == args.epochs - 1)

        train_stats = train_one_epoch(model, train_loader, optimizer,
                                      aux_optimizer, args.metric, device, show_progress=is_log_epoch)
        val_stats   = validate(model, val_loader, args.metric, device, show_progress=is_log_epoch)
        scheduler.step()

        elapsed = time.time() - t0
        history['train'].append(train_stats)
        history['val'].append(val_stats)

        if is_log_epoch:
            print(f"[{epoch+1:4d}/{args.epochs}] "
                  f"Loss:{train_stats['loss']:.4f}  "
                  f"bpp:{train_stats['bpp']:.4f}  "
                  f"R_motion:{train_stats.get('R_motion', 0):.4f}  "
                  f"R_residual:{train_stats.get('R_residual', 0):.4f}  "
                  f"PSNR:{train_stats['psnr']:.2f}  │  "
                  f"V-Loss:{val_stats['loss']:.4f}  "
                  f"V-bpp:{val_stats['bpp']:.4f}  "
                  f"V-PSNR:{val_stats['psnr']:.2f}  "
                  f"V-SSIM:{val_stats['ssim']:.4f}  "
                  f"({elapsed:.1f}s)")

        if val_stats['psnr'] > best_val_psnr:
            best_val_psnr = val_stats['psnr']
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_psnr': best_val_psnr,
                'history': history,
                'args': vars(args),
            }, os.path.join(ckpt_dir, 'best.pth'))
            if is_log_epoch:
                print(f"  ✓ best model saved (PSNR={best_val_psnr:.2f} dB)")

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'history': history,
            'args': vars(args),
        }, os.path.join(ckpt_dir, 'latest.pth'))

    with open(os.path.join(ckpt_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best Val PSNR: {best_val_psnr:.2f} dB")
    print(f"Checkpoint dir: {ckpt_dir}/")

if __name__ == '__main__':
    main()