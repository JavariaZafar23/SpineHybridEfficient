# SpineHybridEfficient

**Frozen Self-Supervised Vision Transformers with Dual Encoders for Single-Stage 3D Vertebrae Instance Segmentation**

> MSc Data Science with Advanced Research — University of Hertfordshire (MSDS24021), 2026  
> Submitted to: *Biomedical Signal Processing and Control* (Elsevier)

---

## Architecture

```
CT Volume (128³)
        │
   ┌────┴────┐
   │         │
[3DINO      [ResNet50
ViT-Large   trainable]
last 2 blk       │
unfrozen]    skip1 (256ch @ 16³)
   │         skip2 (512ch @  8³)
   └────┬────┘
        │
   Fusion @ 4³ bottleneck
        │
   ┌────▼─────────────────┐
   │  Decoder             │
   │  RTUpBlock ×2        │
   │  UpBlock   ×3        │
   │  (NestedUNet + RT)   │
   └──────────┬───────────┘
              │
       SegHead → 26 classes
```

---

## Results on VerSe 2019+2020

| Method | Val DSC | Trainable Params | Stage |
|---|---|---|---|
| UNETR | 0.3926 | 86M | Single |
| ViT-Adapter-UNETR | 0.4160 | 43M | Single |
| nnU-Net | 0.71 | ~32M | Single |
| **SpineHybridEfficient (ours)** | **0.7845** | **31M** | **Single** |

---

## Dataset

VerSe 2019 + VerSe 2020: https://github.com/anjany/verse  
284 CT scans | 26 classes (background + C1–S1)

---

## Repository Structure

```
SpineHybridEfficient/
├── src/
│   ├── models/
│   │   └── model.py             ← SpineHybridEfficient architecture
│   ├── train/
│   │   └── train.py             ← training script
│   ├── inference/
│   │   └── infer.py             ← 256³ sliding window inference
│   ├── preprocess/
│   │   └── preprocess.py        ← VerSe19+20 preprocessing
│   └── utils/
│       ├── dataset.py           ← SpineNpyDataset + augmentation
│       ├── losses.py            ← combined loss functions
│       └── metrics.py           ← Dice + identification rate
├── configs/
│   └── config.yaml
├── results/
│   └── logs/
├── paper/
│   ├── main.tex
│   └── references.bib
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/ansmalik67/SpineHybridEfficient
cd SpineHybridEfficient
pip install -r requirements.txt
```

### 3DINO Encoder

Required but not included. Download from: https://github.com/YtongXie/3DINO  
Place weights at: `path/to/3DINO/weights/3dino_vit_weights.pth`  
Update `dino_weights` in `configs/config.yaml`.

---

## Preprocessing

```bash
# Update paths in src/preprocess/preprocess.py first
python src/preprocess/preprocess.py
```

Produces `images/` and `labels/` as `.npy` files (~30–60 min on CPU).

---

## Training

```bash
# Inside tmux to survive disconnections
tmux new -s train
python src/train/train.py 2>&1 | tee results/logs/train.log
# Ctrl+b then d  →  detach (training keeps running)
# tmux attach -t train  →  reconnect later
```

Training resumes automatically from the last checkpoint.

---

## Inference at 256³

```bash
python src/inference/infer.py \
    --weights results/SpineHybridEfficient/best_model.pth \
    --input   /path/to/preprocessed/images \
    --labels  /path/to/preprocessed_256/labels \
    --cases   /path/to/combined_val.txt \
    --output  results/predictions \
    --roi     128 \
    --overlap 0.5
```

---

## Citation

```bibtex
@article{riaz2026spinehybrid,
  title   = {SpineHybridEfficient: Frozen Self-Supervised Vision Transformers
             with Dual Encoders for 3D Vertebrae Instance Segmentation},
  author  = {Riaz, Ans},
  journal = {Biomedical Signal Processing and Control (under review)},
  year    = {2026}
}
```

---

## License

MIT License
