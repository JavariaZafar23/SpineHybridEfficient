"""
train.py
========
Training script for SpineHybridEfficient.

Usage:
    python src/train/train.py

Resumes automatically from the last checkpoint if one exists.
Run inside tmux to keep training alive across disconnections:

    tmux new -s train
    python src/train/train.py 2>&1 | tee results/logs/train.log
    # Ctrl+b then d  →  detach (training keeps running)
    # tmux attach -t train  →  reconnect later
"""

import os
import sys
import gc
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import psutil

# Add repo root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, '/home/muhammadfaisal/Documents/Jenv/3DINO')

os.environ["XFORMERS_DISABLED"]       = "1"
os.environ["CUDA_VISIBLE_DEVICES"]    = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.models.model   import build_model
from src.utils.dataset  import SpineNpyDataset, load_cases
from src.utils.losses   import combined_loss, dice_score

# ============================================================
# CONFIG  —  update paths to match your machine
# ============================================================
PREPROCESSED_DIR = '/home/Documents/Jenv/data/preprocessed'
RESULTS_DIR      = '/home/Documents/Jenv/results/SpineHybridEfficient'
DINO_WEIGHTS     = '/home/Documents/Jenv/3DINO/weights/3dino_vit_weights.pth'

IMAGE_SIZE    = 128     # training resolution
NUM_EPOCHS    = 300
LR            = 1e-4
NUM_CLASSES   = 26
BATCH_SIZE    = 1
WARMUP_EPOCHS = 10

os.makedirs(RESULTS_DIR, exist_ok=True)
BEST   = f'{RESULTS_DIR}/best_model.pth'
LATEST = f'{RESULTS_DIR}/latest_checkpoint.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*60}")
print(f"SpineHybridEfficient — VerSe 2019+2020 Training")
print(f"{'='*60}")
print(f"Device : {device} | GPU: {torch.cuda.get_device_name(0)}")
print(f"RAM    : {psutil.virtual_memory().total/1e9:.1f} GB")
print(f"{'='*60}\n", flush=True)

# ============================================================
# DATA
# ============================================================
print("Loading cases...")
train_cases = (
    load_cases(f'{PREPROCESSED_DIR}/verse19_train.txt') +
    load_cases(f'{PREPROCESSED_DIR}/verse19_test.txt')  +
    load_cases(f'{PREPROCESSED_DIR}/verse20_train.txt') +
    load_cases(f'{PREPROCESSED_DIR}/verse20_test.txt'))
val_cases = (
    load_cases(f'{PREPROCESSED_DIR}/verse19_val.txt') +
    load_cases(f'{PREPROCESSED_DIR}/verse20_val.txt'))

print(f"Train: {len(train_cases)} | Val: {len(val_cases)}", flush=True)

img_dir = f'{PREPROCESSED_DIR}/images'
lbl_dir = f'{PREPROCESSED_DIR}/labels'

train_loader = DataLoader(
    SpineNpyDataset(train_cases, img_dir, lbl_dir, augment=True,  image_size=IMAGE_SIZE),
    batch_size=BATCH_SIZE, shuffle=True,  num_workers=1,
    pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(
    SpineNpyDataset(val_cases, img_dir, lbl_dir, augment=False, image_size=IMAGE_SIZE),
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

print(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches\n", flush=True)

# ============================================================
# MODEL
# ============================================================
model, encoder = build_model(
    dino_weights_path     = DINO_WEIGHTS,
    device                = device,
    num_classes           = NUM_CLASSES,
    target_size           = IMAGE_SIZE,
    unfreeze_last_n_blocks = 2)   # unfreeze last 2 blocks → best Dice

# ============================================================
# OPTIMIZER & SCHEDULER
# ============================================================
dino_params  = [p for p in encoder.parameters() if p.requires_grad]
other_params = [p for p in model.parameters()
                if p.requires_grad and not any(p is d for d in dino_params)]

optimizer = torch.optim.AdamW([
    {'params': dino_params,  'lr': LR * 0.1},   # lower LR for DINO fine-tune
    {'params': other_params, 'lr': LR},
], weight_decay=5e-4, betas=(0.9, 0.999))

scheduler         = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)
scaler            = torch.amp.GradScaler('cuda')
best_val_dice     = -1.0
start_epoch       = 0

# ============================================================
# CHECKPOINT  —  atomic write (crash-safe)
# ============================================================
def save_checkpoint(epoch, is_best=False):
    global best_val_dice
    ckpt = {
        'epoch':         epoch,
        'model':         model.state_dict(),
        'optimizer':     optimizer.state_dict(),
        'scheduler':     scheduler.state_dict(),
        'scaler':        scaler.state_dict(),
        'best_val_dice': best_val_dice,
    }
    tmp = LATEST + '.tmp'
    try:
        torch.save(ckpt, tmp)
        os.replace(tmp, LATEST)   # atomic — safe if machine crashes mid-write
        print(f"  -> Checkpoint saved (epoch {epoch+1})", flush=True)
    except Exception as e:
        print(f"  WARNING: save failed: {e}", flush=True)
    if is_best:
        try:
            torch.save(model.state_dict(), BEST)
            print(f"  -> Best model saved  (dice={best_val_dice:.4f})", flush=True)
        except Exception as e:
            print(f"  WARNING: best save failed: {e}", flush=True)


if os.path.exists(LATEST):
    print("Resuming from checkpoint...")
    try:
        c             = torch.load(LATEST, map_location=device, weights_only=False)
        model.load_state_dict(c['model'])
        optimizer.load_state_dict(c['optimizer'])
        scaler.load_state_dict(c['scaler'])
        start_epoch   = c['epoch'] + 1
        best_val_dice = c.get('best_val_dice', -1.0)
        if 'scheduler' in c:
            scheduler.load_state_dict(c['scheduler'])
        print(f"  Resumed epoch {start_epoch} | Best dice: {best_val_dice:.4f}", flush=True)
    except Exception as e:
        print(f"  Load failed: {e}  —  starting fresh", flush=True)
else:
    print("No checkpoint found — starting fresh\n", flush=True)

print(f"Epochs  : {start_epoch} → {NUM_EPOCHS}")
print(f"Free GPU: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB\n", flush=True)

# ============================================================
# TRAIN ONE EPOCH
# ============================================================
def train_one_epoch(epoch):
    model.train()
    encoder.train()
    total_loss, dice_vals, skipped, processed = 0.0, [], 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d} [Train]",
                unit="batch", leave=False, bar_format="{l_bar}{bar:20}{r_bar}")

    for i, (x, y, name) in enumerate(pbar):
        try:
            if i % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if torch.isnan(x).any() or y.max() >= NUM_CLASSES or y.min() < 0:
                skipped += 1
                continue

            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss   = combined_loss(logits, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            processed  += 1

            # Dice logging every 10 batches
            if i % 10 == 0:
                with torch.no_grad():
                    pred  = logits.detach().argmax(dim=1).cpu()
                    y_cpu = y.cpu()
                    for c in range(1, NUM_CLASSES):
                        pc = (pred == c).float()
                        tc = (y_cpu == c).float()
                        if tc.sum() > 0:
                            dice_vals.append(dice_score(pc, tc))

            pbar.set_postfix({
                'loss': f'{total_loss/processed:.4f}',
                'dice': f'{sum(dice_vals)/len(dice_vals):.4f}' if dice_vals else '0',
                'GPU' : f'{torch.cuda.memory_allocated()/1e9:.2f}GB'})
            del logits, x, y, loss

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                pbar.write(f"  OOM skip {name[0]}")
            else:
                pbar.write(f"  SKIP {name[0]}: {str(e)[:60]}")
            skipped += 1
        except Exception as e:
            skipped += 1
            pbar.write(f"  SKIP {name[0]}: {str(e)[:60]}")
            torch.cuda.empty_cache()

    pbar.close()
    torch.cuda.empty_cache()
    gc.collect()
    if skipped:
        print(f"  Skipped {skipped}/{len(train_loader)} batches", flush=True)
    return (total_loss / max(processed, 1),
            float(sum(dice_vals) / len(dice_vals)) if dice_vals else 0.0)


# ============================================================
# VALIDATE  —  3-pass TTA (D-flip + H-flip + original)
# ============================================================
def validate():
    model.eval()
    encoder.eval()
    torch.cuda.empty_cache()
    all_dice, all_losses = [], []

    pbar = tqdm(val_loader, desc="Validating",
                unit="batch", leave=False, bar_format="{l_bar}{bar:20}{r_bar}")

    with torch.no_grad():
        for x, y, name in pbar:
            try:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if torch.isnan(x).any() or y.max() >= NUM_CLASSES:
                    continue

                with torch.amp.autocast('cuda'):
                    logits  = model(x)
                    logits += torch.flip(model(torch.flip(x, [2])), [2])
                    logits += torch.flip(model(torch.flip(x, [3])), [3])
                    logits  = logits / 3.0
                    all_losses.append(combined_loss(logits, y).item())

                pred  = logits.argmax(dim=1).cpu()
                y_cpu = y.cpu()
                bd    = [dice_score((pred == c).float(), (y_cpu == c).float())
                         for c in range(1, NUM_CLASSES)
                         if (y_cpu == c).float().sum() > 0]
                if bd:
                    all_dice.append(float(sum(bd) / len(bd)))
                    pbar.set_postfix({'val_dice': f'{sum(all_dice)/len(all_dice):.4f}'})
                del logits, x, y, pred, y_cpu
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    pbar.write(f"  OOM val {name[0]}")
                else:
                    pbar.write(f"  SKIP {name[0]}: {str(e)[:60]}")
            except Exception as e:
                pbar.write(f"  SKIP {name[0]}: {str(e)[:60]}")
                torch.cuda.empty_cache()

    pbar.close()
    return (float(sum(all_dice)   / len(all_dice))   if all_dice   else 0.0,
            float(sum(all_losses) / len(all_losses)) if all_losses else 0.0)


# ============================================================
# TRAINING LOOP
# ============================================================
print("=" * 60)
print("Starting Training")
print("=" * 60, flush=True)

for epoch in range(start_epoch, NUM_EPOCHS):
    t0         = time.time()
    ram_before = psutil.virtual_memory().used / 1e9

    # LR warmup
    if epoch < WARMUP_EPOCHS:
        wf = (epoch + 1) / WARMUP_EPOCHS
        for i, pg in enumerate(optimizer.param_groups):
            pg['lr'] = (LR * 0.1 if i == 0 else LR) * wf

    avg_loss, train_dice = train_one_epoch(epoch)
    val_dice,  val_loss  = validate()

    # Step scheduler only after warmup
    if epoch >= WARMUP_EPOCHS:
        scheduler.step()

    is_best = val_dice > best_val_dice
    if is_best:
        best_val_dice = val_dice

    ram_used = psutil.virtual_memory().used / 1e9
    elapsed  = time.time() - t0
    cur_lr   = optimizer.param_groups[1]['lr']

    print(f"Epoch {epoch+1:03d} | "
          f"loss: {avg_loss:.4f} | val_loss: {val_loss:.4f} | "
          f"train_dice: {train_dice:.4f} | val_dice: {val_dice:.4f} | "
          f"best: {best_val_dice:.4f} | lr: {cur_lr:.2e} | "
          f"GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB | "
          f"RAM: {ram_used:.1f}GB ({ram_used-ram_before:+.1f}) | "
          f"time: {elapsed:.0f}s",
          flush=True)

    save_checkpoint(epoch, is_best)
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n{'='*60}")
print(f"Training complete!  Best Val Dice: {best_val_dice:.4f}")
print(f"{'='*60}")
