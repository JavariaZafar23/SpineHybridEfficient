"""
losses.py
=========
Loss functions for vertebra segmentation.

Combined loss = 0.3×GDL + 0.3×SDL + 0.1×CE + 0.3×FL
"""

import torch
import torch.nn.functional as F

NUM_CLASSES = 26


def dice_score(pred, target, eps=1e-8):
    """Binary Dice similarity coefficient (scalar)."""
    num = 2 * (pred * target).sum() + eps
    den = pred.sum() + target.sum() + eps
    return (num / den).item()


def dice_per_class_loss(output, target, eps=1e-5):
    """Per-class soft Dice loss — background excluded."""
    probs        = torch.softmax(output, dim=1)
    total, valid = 0.0, 0
    for c in range(1, output.shape[1]):
        p = probs[:, c]
        t = (target == c).float()
        if t.sum() > 0:
            total += 1.0 - (2 * (p * t).sum()) / (p.sum() + t.sum() + eps)
            valid += 1
    return total / max(valid, 1)


def generalized_dice_loss(output, target, eps=1e-5):
    """
    Generalized Dice Loss.
    Memory-safe: class-by-class iteration avoids the full (B,C,N) one-hot tensor.
    """
    probs            = torch.softmax(output, dim=1)
    B                = probs.shape[0]
    num_sum, den_sum = 0.0, 0.0
    for c in range(NUM_CLASSES):
        p       = probs[:, c].reshape(B, -1)
        t       = (target == c).float().reshape(B, -1)
        w       = 1.0 / (t.sum(-1) ** 2 + eps)
        num_sum = num_sum + (w * (p * t).sum(-1)).sum()
        den_sum = den_sum + (w * (p + t).sum(-1)).sum()
    return (1 - 2 * num_sum / (den_sum + eps)) / B


def focal_loss(output, target, gamma=2.0, alpha=0.25):
    """Focal loss to address severe class imbalance."""
    ce    = F.cross_entropy(output, target.long(), reduction='none')
    pt    = torch.exp(-ce)
    return (alpha * (1 - pt) ** gamma * ce).mean()


def combined_loss(output, target):
    """
    Combined segmentation loss:
        0.3 × Generalized Dice
      + 0.3 × Per-class Dice
      + 0.1 × Cross-Entropy
      + 0.3 × Focal
    """
    return (0.3 * generalized_dice_loss(output, target) +
            0.3 * dice_per_class_loss(output, target) +
            0.1 * F.cross_entropy(output, target.long()) +
            0.3 * focal_loss(output, target))
