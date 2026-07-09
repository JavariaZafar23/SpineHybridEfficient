"""
dataset.py
==========
SpineNpyDataset — loads preprocessed .npy CT volumes and labels.
Supports full 3D augmentation pipeline for training.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, map_coordinates


class SpineNpyDataset(Dataset):
    """
    Dataset for VerSe 2019+2020 preprocessed as .npy files.

    Args:
        cases      : list of case name strings (no file extension)
        img_dir    : path to directory of image .npy files
        lbl_dir    : path to directory of label .npy files
        augment    : apply augmentation if True (use for train only)
        image_size : spatial size of loaded volumes
    """
    def __init__(self, cases, img_dir, lbl_dir, augment=False, image_size=128):
        self.cases      = cases
        self.img_dir    = img_dir
        self.lbl_dir    = lbl_dir
        self.augment    = augment
        self.image_size = image_size

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        name = self.cases[idx]
        img  = np.load(f'{self.img_dir}/{name}.npy').astype(np.float32)
        lbl  = np.load(f'{self.lbl_dir}/{name}.npy').astype(np.int64)
        img  = torch.from_numpy(img).unsqueeze(0)   # (1, D, H, W)
        lbl  = torch.from_numpy(lbl)                # (D, H, W)
        if self.augment:
            img, lbl = self._augment(img, lbl)
        return img, lbl, name

    # ------------------------------------------------------------------
    def _elastic_deform(self, img, lbl):
        alpha  = random.uniform(50, 150)
        sigma  = random.uniform(8, 12)
        img_np = img.squeeze(0).numpy()
        lbl_np = lbl.numpy()
        shape  = img_np.shape
        dx = gaussian_filter(np.random.randn(*shape), sigma) * alpha
        dy = gaussian_filter(np.random.randn(*shape), sigma) * alpha
        dz = gaussian_filter(np.random.randn(*shape), sigma) * alpha
        xi, yi, zi = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]),
            np.arange(shape[2]), indexing='ij')
        indices = (np.clip(xi+dx, 0, shape[0]-1).flatten(),
                   np.clip(yi+dy, 0, shape[1]-1).flatten(),
                   np.clip(zi+dz, 0, shape[2]-1).flatten())
        img_def = map_coordinates(img_np, indices, order=1).reshape(shape)
        lbl_def = map_coordinates(lbl_np.astype(np.float32), indices,
                                  order=0).reshape(shape).astype(np.int64)
        return torch.from_numpy(img_def).unsqueeze(0), torch.from_numpy(lbl_def)

    def _augment(self, img, lbl):
        # Random flips
        for axis in [1, 2, 3]:
            if random.random() > 0.5:
                img = torch.flip(img, [axis])
                lbl = torch.flip(lbl, [axis - 1])
        # 90° rotation in axial plane
        if random.random() > 0.7:
            k   = random.randint(1, 3)
            img = torch.rot90(img, k, [2, 3])
            lbl = torch.rot90(lbl, k, [1, 2])
        # Elastic deformation
        if random.random() > 0.7:
            img, lbl = self._elastic_deform(img, lbl)
        # Gaussian noise
        if random.random() > 0.7:
            img = torch.clamp(img + torch.randn_like(img) * 0.02, 0.0, 1.0)
        # Intensity scale
        if random.random() > 0.6:
            img = torch.clamp(img * (1.0 + (random.random() - 0.5) * 0.2), 0.0, 1.0)
        # Brightness shift
        if random.random() > 0.7:
            img = torch.clamp(img + random.uniform(-0.1, 0.1), 0.0, 1.0)
        # Gamma correction
        if random.random() > 0.7:
            img = torch.clamp(img ** random.uniform(0.7, 1.5), 0.0, 1.0)
        # Gaussian blur
        if random.random() > 0.7:
            sigma  = random.uniform(0.5, 1.5)
            img_np = gaussian_filter(img.squeeze(0).numpy(), sigma=sigma)
            img    = torch.clamp(
                torch.from_numpy(img_np).unsqueeze(0).float(), 0.0, 1.0)
        # Simulate low resolution
        if random.random() > 0.8:
            scale      = random.uniform(0.5, 0.8)
            small_size = max(32, int(self.image_size * scale))
            img = F.interpolate(img.unsqueeze(0), size=(small_size,) * 3,
                                mode='trilinear', align_corners=False)
            img = F.interpolate(img, size=(self.image_size,) * 3,
                                mode='trilinear', align_corners=False).squeeze(0)
            img = torch.clamp(img, 0.0, 1.0)
        return img, lbl


def load_cases(filepath):
    """Read a plain-text list of case names (one per line)."""
    with open(filepath) as f:
        return [line.strip() for line in f if line.strip()]
