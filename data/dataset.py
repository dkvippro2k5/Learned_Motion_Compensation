"""
data/dataset.py — Dataset utilities for Learned Motion Compensation.

Provides:
  1. FramePairDataset  — loads (ref, cur) frame pairs from ref/ and cur/ dirs
  2. get_dataloaders() — builds train/val DataLoaders and splits them

The (ref, cur) pairs are produced from the raw .mp4 clips by
extract_custom_videos.py (this module does not read video files itself).
"""

import os
import glob
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader


class FramePairDataset(Dataset):
    """Load (ref, cur) frame pairs from ref/ and cur/ subdirectories
    (pre-extracted consecutive frames)."""
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


def get_dataloaders(data_root='data/real', batch_size=8,
                    height=256, width=448, val_split=0.1,
                    num_workers=0):
    """Build train/val DataLoaders from (ref, cur) pairs at data_root.
    Run extract_custom_videos.py first to create them."""
    ref_dir = os.path.join(data_root, 'ref')
    if not os.path.exists(ref_dir):
        raise FileNotFoundError(
            f"No frames at {ref_dir}. Run: python extract_custom_videos.py")

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
