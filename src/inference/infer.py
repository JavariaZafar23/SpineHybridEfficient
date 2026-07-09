"""
infer.py
========
Sliding-window inference at 256³ for SpineHybridEfficient.

The model is trained at 128³ but inference runs over the full 256³
volume using MONAI's Gaussian-weighted sliding window, which eliminates
seam artefacts at patch boundaries.

Usage:
    python src/inference/infer.py \
        --weights results/SpineHybridEfficient/best_model.pth \
        --input   /path/to/preprocessed_256/images \
        --labels  /path/to/preprocessed_256/labels \
        --cases   /path/to/combined_val.txt \
        --output  results/predictions
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, '/home/muhammadfaisal/Documents/Jenv/3DINO')

os.environ["XFORMERS_DISABLED"]       = "1"
os.environ["CUDA_VISIBLE_DEVICES"]    = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from monai.inferers      import sliding_window_inference
from src.models.model    import build_model
from src.utils.metrics   import mean_dice, identification_rate, per_class_dice

DINO_WEIGHTS = '/home/muhammadfaisal/Documents/Jenv/3DINO/weights/3dino_vit_weights.pth'
NUM_CLASSES  = 26


def load_model(weights_path, device, roi_size=128):
    model, _ = build_model(
        dino_weights_path      = DINO_WEIGHTS,
        device                 = device,
        num_classes            = NUM_CLASSES,
        target_size            = roi_size,
        unfreeze_last_n_blocks = 0)   # frozen at inference

    state = torch.load(weights_path, map_location=device, weights_only=False)
    if 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    model.eval()
    print(f"Weights loaded from: {weights_path}\n")
    return model


def infer_volume(model, img_path, device, roi_size=128, overlap=0.5):
    """
    Run sliding-window inference on one 256³ .npy volume.

    Returns:
        pred : (D, H, W) int64 numpy array
    """
    img = np.load(img_path).astype(np.float32)
    vol = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)

    with torch.no_grad(), torch.amp.autocast('cuda'):
        logits = sliding_window_inference(
            inputs        = vol,
            roi_size      = (roi_size,) * 3,
            sw_batch_size = 1,
            predictor     = model,
            overlap       = overlap,
            mode          = 'gaussian')   # Gaussian-weighted boundary blending

    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True,  help='Path to best_model.pth')
    parser.add_argument('--input',   required=True,  help='Directory of 256³ image .npy files')
    parser.add_argument('--labels',  default=None,   help='Directory of label .npy files (optional, for evaluation)')
    parser.add_argument('--cases',   required=True,  help='Text file of case names')
    parser.add_argument('--output',  required=True,  help='Output directory for predictions')
    parser.add_argument('--roi',     type=int,   default=128,  help='Sliding window patch size')
    parser.add_argument('--overlap', type=float, default=0.5,  help='Sliding window overlap')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    model = load_model(args.weights, device, args.roi)

    with open(args.cases) as f:
        cases = [l.strip() for l in f if l.strip()]

    print(f"Running inference on {len(cases)} cases  "
          f"(roi={args.roi}³  overlap={args.overlap})\n")

    all_dice, all_idr = [], []

    for i, name in enumerate(cases):
        img_path = os.path.join(args.input, f'{name}.npy')
        out_path = os.path.join(args.output, f'{name}_pred.npy')

        if not os.path.exists(img_path):
            print(f"  [{i+1}/{len(cases)}] SKIP (not found): {name}")
            continue

        pred = infer_volume(model, img_path, device, args.roi, args.overlap)
        np.save(out_path, pred)

        if args.labels:
            lbl_path = os.path.join(args.labels, f'{name}.npy')
            if os.path.exists(lbl_path):
                lbl = np.load(lbl_path).astype(np.int64)
                md  = mean_dice(pred, lbl)
                idr = identification_rate(pred, lbl)
                all_dice.append(md)
                all_idr.append(idr)
                print(f"  [{i+1:03d}/{len(cases)}] {name:30s} "
                      f"Dice={md:.4f}  ID-rate={idr:.4f}")
            else:
                print(f"  [{i+1:03d}/{len(cases)}] {name}  (saved, no label)")
        else:
            print(f"  [{i+1:03d}/{len(cases)}] {name}  (saved)")

    if all_dice:
        print(f"\n{'='*55}")
        print(f"Mean Dice          : {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
        print(f"ID Rate            : {np.mean(all_idr):.4f} ± {np.std(all_idr):.4f}")
        print(f"Cases evaluated    : {len(all_dice)} / {len(cases)}")
        print(f"{'='*55}")

    print(f"\nPredictions saved to: {args.output}")


if __name__ == '__main__':
    main()
