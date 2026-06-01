"""
data/dataset.py — Dataset utilities for Learned Motion Compensation.

Provides:
  1. generate_synthetic_frames() — auto-generated moving-shape pairs (no download)
  2. FramePairDataset          — loads (ref, cur) pairs from ref/ and cur/ dirs
  3. Vimeo90KDataset           — Vimeo-90K septuplet format (DVC/OpenDVC training)
  4. get_dataloaders()         — auto-selects the right dataset and splits train/val

To turn raw .mp4 clips into the ref/cur layout used by FramePairDataset, run
prepare_real_dataset.py (this module does not read video files itself).

Vimeo-90K format expected at:
    vimeo_septuplet/sequences/<clip_id>/<seq_id>/im1.png ... im7.png
Each 7-frame clip yields consecutive pairs (im1, im2), ..., (im6, im7).
"""

import os
import glob
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

def generate_synthetic_frames(num_pairs=300, height=128, width=128,
                               save_dir='data/synthetic'):
    """
    Generate synthetic frame pairs with moving objects.
    Includes translation, zoom, rotation — similar to natural video motion.
    """
    os.makedirs(os.path.join(save_dir, 'ref'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'cur'), exist_ok=True)
    rng = np.random.RandomState(2024)

    for i in range(num_pairs):
        frame_ref = np.zeros((height, width, 3), np.uint8)
        frame_cur = np.zeros((height, width, 3), np.uint8)
        bg = rng.randint(20, 70, 3).tolist()
        frame_ref[:] = bg
        frame_cur[:] = bg

        for _ in range(rng.randint(2, 6)):
            dx = rng.randint(-12, 13)
            dy = rng.randint(-12, 13)
            color = rng.randint(60, 255, 3).tolist()
            shape = rng.randint(0, 3)

            if shape == 0:
                x1 = rng.randint(5, width - 35)
                y1 = rng.randint(5, height - 35)
                w  = rng.randint(15, min(40, width - x1 - 5))
                h  = rng.randint(15, min(40, height - y1 - 5))
                cv2.rectangle(frame_ref, (x1, y1), (x1+w, y1+h), color, -1)
                x2, y2 = np.clip(x1+dx, 0, width-w-1), np.clip(y1+dy, 0, height-h-1)
                cv2.rectangle(frame_cur, (x2, y2), (x2+w, y2+h), color, -1)
            elif shape == 1:
                cx = rng.randint(15, width - 15)
                cy = rng.randint(15, height - 15)
                r = rng.randint(8, 18)
                cv2.circle(frame_ref, (cx, cy), r, color, -1)
                cx2 = int(np.clip(cx+dx, r, width-r-1))
                cy2 = int(np.clip(cy+dy, r, height-r-1))
                cv2.circle(frame_cur, (cx2, cy2), r, color, -1)
            else:
                cx = rng.randint(20, width - 20)
                cy = rng.randint(20, height - 20)
                a, b = rng.randint(10, 22), rng.randint(7, 18)
                ang = rng.randint(0, 180)
                cv2.ellipse(frame_ref, (cx, cy), (a, b), ang, 0, 360, color, -1)
                cx2 = int(np.clip(cx+dx, a, width-a-1))
                cy2 = int(np.clip(cy+dy, b, height-b-1))
                cv2.ellipse(frame_cur, (cx2, cy2), (a, b), ang, 0, 360, color, -1)

        noise = rng.randint(-6, 7, frame_ref.shape).astype(np.int16)
        frame_cur = np.clip(frame_cur.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(save_dir, 'ref', f'{i:05d}.png'), frame_ref)
        cv2.imwrite(os.path.join(save_dir, 'cur', f'{i:05d}.png'), frame_cur)

    print(f"Generated {num_pairs} synthetic frame pairs → {save_dir}/")

class FramePairDataset(Dataset):
    """
    Load (ref, cur) frame pairs from ref/ and cur/ subdirectories.
    Used with synthetic data or any pre-extracted pairs.
    """
    def __init__(self, root, height=128, width=128, augment=False):
        self.height, self.width = height, width
        self.augment = augment
        self.ref_paths = sorted(glob.glob(os.path.join(root, 'ref', '*.png')))
        self.cur_paths = sorted(glob.glob(os.path.join(root, 'cur', '*.png')))
        assert len(self.ref_paths) > 0, f"No frames in {root}/ref/"
        assert len(self.ref_paths) == len(self.cur_paths)

    def __len__(self):
        return len(self.ref_paths)

    def _load(self, path):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.width, self.height))
        return img.astype(np.float32) / 255.0

    def __getitem__(self, idx):
        ref = self._load(self.ref_paths[idx])
        cur = self._load(self.cur_paths[idx])
        if self.augment:
            if np.random.rand() < 0.5:
                ref, cur = ref[:, ::-1].copy(), cur[:, ::-1].copy()
            if np.random.rand() < 0.5:
                ref, cur = ref[::-1].copy(), cur[::-1].copy()
            f = np.random.uniform(0.85, 1.15)
            ref = np.clip(ref * f, 0, 1).astype(np.float32)
            cur = np.clip(cur * f, 0, 1).astype(np.float32)
        return (torch.from_numpy(ref.transpose(2, 0, 1)),
                torch.from_numpy(cur.transpose(2, 0, 1)))

class Vimeo90KDataset(Dataset):
    """
    Vimeo-90K septuplet dataset (same format as used by DVC/OpenDVC).

    Download: http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip (82GB)

    Directory structure:
        vimeo_septuplet/sequences/<00001>/<0001>/im1.png ... im7.png

    Each 7-frame clip gives 6 consecutive pairs: (im1,im2), ..., (im6,im7).
    """
    def __init__(self, root, height=256, width=448, augment=False, max_clips=None):
        self.height, self.width = height, width
        self.augment = augment
        self.pairs = []

        seq_dir = os.path.join(root, 'sequences')
        if not os.path.exists(seq_dir):
            raise FileNotFoundError(
                f"Vimeo-90K not found at {seq_dir}\n"
                "Download from http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip"
            )

        clip_dirs = sorted(glob.glob(os.path.join(seq_dir, '*', '*')))
        if max_clips:
            clip_dirs = clip_dirs[:max_clips]

        for clip_dir in clip_dirs:
            frames = sorted(glob.glob(os.path.join(clip_dir, 'im*.png')))
            for i in range(len(frames) - 1):
                self.pairs.append((frames[i], frames[i + 1]))

        print(f"Vimeo-90K: {len(clip_dirs)} clips, {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ref_path, cur_path = self.pairs[idx]

        ref = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
        cur = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2RGB)
        
        H, W = ref.shape[:2]
        
        if self.augment and (H > self.height and W > self.width):

            y = np.random.randint(0, H - self.height + 1)
            x = np.random.randint(0, W - self.width + 1)
            ref = ref[y:y+self.height, x:x+self.width]
            cur = cur[y:y+self.height, x:x+self.width]
        else:

            if (H, W) != (self.height, self.width):
                ref = cv2.resize(ref, (self.width, self.height))
                cur = cv2.resize(cur, (self.width, self.height))
                
        ref = ref.astype(np.float32) / 255.0
        cur = cur.astype(np.float32) / 255.0

        if self.augment and np.random.rand() < 0.5:
            ref, cur = ref[:, ::-1].copy(), cur[:, ::-1].copy()
            
        return (torch.from_numpy(ref.transpose(2, 0, 1)),
                torch.from_numpy(cur.transpose(2, 0, 1)))

def get_dataloaders(data_root='data/synthetic', batch_size=8,
                    height=128, width=128, val_split=0.1,
                    num_workers=0):
    """
    Build train/val DataLoaders. Auto-generates synthetic data if needed.
    """
    if os.path.exists(os.path.join(data_root, 'sequences')):
        print(f"Detected Vimeo-90K structure at {data_root} — using Vimeo90KDataset.")
        full_ds = Vimeo90KDataset(data_root, height, width, augment=True)
    else:
        if not os.path.exists(os.path.join(data_root, 'ref')):
            print(f"Data not found at {data_root} — generating synthetic...")
            generate_synthetic_frames(300, height, width, data_root)

        full_ds = FramePairDataset(data_root, height, width, augment=True)

    n_val   = max(1, int(len(full_ds) * val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)

    print(f"DataLoaders: {n_train} train / {n_val} val, batch={batch_size}")
    return train_loader, val_loader