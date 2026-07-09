"""
metrics.py
==========
Evaluation metrics for vertebra instance segmentation.
"""

import numpy as np
import torch

NUM_CLASSES = 26


def per_class_dice(pred, target, eps=1e-8):
    """
    Per-class Dice scores, background (class 0) excluded.

    Returns:
        dict {class_id: dice_score} for classes present in target
    """
    if isinstance(pred,   torch.Tensor): pred   = pred.cpu().numpy()
    if isinstance(target, torch.Tensor): target = target.cpu().numpy()
    scores = {}
    for c in range(1, NUM_CLASSES):
        p = (pred   == c).astype(np.float32)
        t = (target == c).astype(np.float32)
        if t.sum() == 0:
            continue
        scores[c] = float((2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps))
    return scores


def mean_dice(pred, target, eps=1e-8):
    """Mean Dice over all vertebra classes present in target."""
    scores = per_class_dice(pred, target, eps)
    return float(np.mean(list(scores.values()))) if scores else 0.0


