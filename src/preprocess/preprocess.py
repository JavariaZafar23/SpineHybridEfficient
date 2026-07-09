"""
preprocess.py
=============
Preprocess VerSe 2019 + VerSe 2020 to .npy files.

Pipeline per case:
  1. Read NIfTI
  2. Reorient → RAS
  3. Resample → 1 mm isotropic
  4. HU clip [-1000, 800] + normalize [0, 1]
  5. Foreground crop (10 voxel margin)
  6. Resize → IMAGE_SIZE³
  7. Save float32 image + uint8 label

Usage:
    python src/preprocess/preprocess.py

Update VERSE19_DIR, VERSE20_DIR, and OUTPUT_DIR before running.
"""

import os
import numpy as np
import SimpleITK as sitk

# ============================================================
# PATHS — update to your machine
# ============================================================
OUTPUT_DIR  = '/home/Documents/Jenv/data/preprocessed'
VERSE19_DIR = '/home/Documents/Jenv/data/verse19'
VERSE20_DIR = '/home/Documents/Jenv/data/verse20'

IMAGE_SIZE     = 128            # set to 256 for high-res inference data
TARGET_SPACING = [1.0, 1.0, 1.0]
BLACKLIST      = {'sub-verse588', 'sub-verse631', 'sub-verse582', 'sub-verse602'}

os.makedirs(f'{OUTPUT_DIR}/images', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/labels', exist_ok=True)

print(f"Output : {OUTPUT_DIR}")
print(f"Size   : {IMAGE_SIZE}³  |  Spacing: {TARGET_SPACING}\n", flush=True)


# ============================================================
# CORE FUNCTIONS
# ============================================================
def reorient_to_ras(img):
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation('RAS')
    return orienter.Execute(img)


def resample(img, new_spacing, is_label=False):
    orig_sp   = img.GetSpacing()
    orig_size = img.GetSize()
    new_size  = [int(round(s * osp / nsp))
                 for s, osp, nsp in zip(orig_size, orig_sp, new_spacing)]
    interp    = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(new_spacing); r.SetSize(new_size)
    r.SetOutputDirection(img.GetDirection()); r.SetOutputOrigin(img.GetOrigin())
    r.SetTransform(sitk.Transform()); r.SetDefaultPixelValue(0)
    r.SetInterpolator(interp)
    return r.Execute(img)


def crop_foreground(img_arr, lbl_arr, margin=10):
    nz = np.where(lbl_arr > 0)
    if len(nz[0]) == 0:
        return img_arr, lbl_arr
    mins = [max(0, n.min() - margin) for n in nz]
    maxs = [min(s, n.max() + margin) for s, n in zip(lbl_arr.shape, nz)]
    return (img_arr[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]],
            lbl_arr[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]])


def resize_volume(arr, size, is_label=False):
    si       = sitk.GetImageFromArray(arr)
    new_sp   = [float(o) / float(size) for o in si.GetSize()]
    interp   = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    r = sitk.ResampleImageFilter()
    r.SetSize([size]*3); r.SetOutputSpacing(new_sp)
    r.SetOutputDirection(si.GetDirection()); r.SetOutputOrigin(si.GetOrigin())
    r.SetTransform(sitk.Transform()); r.SetDefaultPixelValue(0)
    r.SetInterpolator(interp)
    return sitk.GetArrayFromImage(r.Execute(si))


def preprocess_pair(img_path, lbl_path, case_name):
    out_img = f'{OUTPUT_DIR}/images/{case_name}.npy'
    out_lbl = f'{OUTPUT_DIR}/labels/{case_name}.npy'
    if os.path.exists(out_img) and os.path.exists(out_lbl):
        return True, "already exists"
    try:
        img_s = reorient_to_ras(sitk.ReadImage(img_path))
        lbl_s = reorient_to_ras(sitk.ReadImage(lbl_path))
        img_s = resample(img_s, TARGET_SPACING, is_label=False)
        lbl_s = resample(lbl_s, TARGET_SPACING, is_label=True)
        img   = sitk.GetArrayFromImage(img_s).astype(np.float32)
        lbl   = sitk.GetArrayFromImage(lbl_s).astype(np.uint8)
        img   = np.clip((img - (-1000)) / (800 - (-1000)), 0.0, 1.0)
        if lbl.max() == 0:   return False, "empty label"
        if lbl.max() > 25:   return False, f"label > 25: {lbl.max()}"
        img, lbl = crop_foreground(img, lbl, margin=10)
        if any(s == 0 for s in img.shape): return False, "zero size after crop"
        img = resize_volume(img, IMAGE_SIZE, is_label=False).astype(np.float32)
        lbl = resize_volume(lbl, IMAGE_SIZE, is_label=True).astype(np.uint8)
        np.save(out_img, img); np.save(out_lbl, lbl)
        return True, f"shape={img.shape} labels={np.unique(lbl).tolist()}"
    except Exception as e:
        return False, str(e)[:120]


# ============================================================
# FILE FINDERS
# ============================================================
def find_ct(d):
    f = [x for x in os.listdir(d) if x.endswith('_ct.nii.gz')]
    return f[0] if f else None

def find_mask(d):
    f = [x for x in os.listdir(d) if x.endswith('.nii.gz') and '_msk' in x]
    seg = [x for x in f if '_seg-' in x]
    return seg[0] if seg else (f[0] if f else None)

def collect(rawdata_dir, deriv_dir, split_name):
    if not os.path.isdir(rawdata_dir):
        print(f"  SKIP (not found): {rawdata_dir}"); return []
    pairs = []
    for subj in sorted(os.listdir(rawdata_dir)):
        if subj in BLACKLIST: continue
        rd = os.path.join(rawdata_dir, subj)
        dd = os.path.join(deriv_dir,   subj)
        if not os.path.isdir(rd): continue
        ct   = find_ct(rd);   mask = find_mask(dd) if os.path.isdir(dd) else None
        if not ct or not mask: continue
        pairs.append((os.path.join(rd, ct), os.path.join(dd, mask), subj))
    print(f"  {split_name}: {len(pairs)} pairs")
    return pairs

def run_split(pairs, split_name, txt_name):
    print(f"\n{'='*50}\n  {split_name}  ({len(pairs)} cases)\n{'='*50}", flush=True)
    good, bad = [], []
    for i, (img_p, lbl_p, name) in enumerate(pairs):
        ok, msg = preprocess_pair(img_p, lbl_p, name)
        print(f"  [{i+1:03d}/{len(pairs)}] {'OK ' if ok else 'BAD'}: {name} -- {msg}", flush=True)
        (good if ok else bad).append(name)
    if bad: print(f"  Failed: {bad}")
    with open(f'{OUTPUT_DIR}/{txt_name}', 'w') as f:
        for n in good: f.write(n + '\n')
    print(f"  Saved: {OUTPUT_DIR}/{txt_name}", flush=True)
    return good, bad


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    results = {}

    # VerSe19 — dataset-verse19training / validation / test
    print("="*50); print("VERSE 19"); print("="*50)
    for folder, split, txt in [
        ('dataset-verse19training',   'verse19_train', 'verse19_train.txt'),
        ('dataset-verse19validation', 'verse19_val',   'verse19_val.txt'),
        ('dataset-verse19test',       'verse19_test',  'verse19_test.txt'),
    ]:
        root = os.path.join(VERSE19_DIR, folder)
        if not os.path.isdir(root): print(f"  SKIP: {root}"); continue
        pairs = collect(os.path.join(root, 'rawdata'),
                        os.path.join(root, 'derivatives'), split)
        if pairs:
            results[split] = run_split(pairs, split, txt)

    # VerSe20 — train + test (no validationdata folder)
    print("\n"+"="*50); print("VERSE 20"); print("="*50)
    for rd, dd, split, txt in [
        (os.path.join(VERSE20_DIR, 'rawdata'),
         os.path.join(VERSE20_DIR, 'derivatives'),
         'verse20_train', 'verse20_train.txt'),
        (os.path.join(VERSE20_DIR, 'testingdata', 'rawdata'),
         os.path.join(VERSE20_DIR, 'testingdata', 'derivatives'),
         'verse20_test', 'verse20_test.txt'),
    ]:
        pairs = collect(rd, dd, split)
        if pairs:
            results[split] = run_split(pairs, split, txt)

    # Write combined lists
    train_splits = ['verse19_train', 'verse19_test', 'verse20_train', 'verse20_test']
    val_splits   = ['verse19_val']
    train_all = [n for s in train_splits if s in results for n in results[s][0]]
    val_all   = [n for s in val_splits   if s in results for n in results[s][0]]
    with open(f'{OUTPUT_DIR}/combined_train.txt', 'w') as f:
        for n in train_all: f.write(n + '\n')
    with open(f'{OUTPUT_DIR}/combined_val.txt', 'w') as f:
        for n in val_all:   f.write(n + '\n')

    # Summary
    print("\n"+"="*50); print("SUMMARY"); print("="*50)
    tg = tb = 0
    for s, (g, b) in results.items():
        print(f"  {s:<22}: {len(g):>3} OK | {len(b):>2} failed")
        tg += len(g); tb += len(b)
    print(f"  {'TOTAL':<22}: {tg:>3} OK | {tb:>2} failed")
    print(f"  combined_train: {len(train_all)}  |  combined_val: {len(val_all)}")
    print(f"\nOutput: {OUTPUT_DIR}")
