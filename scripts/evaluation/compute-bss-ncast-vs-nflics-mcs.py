import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import scipy.ndimage as nd

# read arguments
lead_time = sys.argv[1]
target_hour = sys.argv[2]

# directories
base_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/predictions/ncast-nflics-full/t{lead_time}"
output_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/ncast-nflics-full/bss/1"
os.makedirs(output_dir, exist_ok=True)

H, W = 512, 512

# 25 km corresponds to a 9 pixel neighbourhood
L_pixels = 1

# initialise accumulators
bs_model_sum = np.zeros((H, W), dtype=np.float64)
bs_ref_sum   = np.zeros((H, W), dtype=np.float64)
count        = np.zeros((H, W), dtype=np.int32)

# count connected convective cores in a binary mask
def count_cores(mask):
    labelled, n = nd.label(mask > 0.5)
    return n

# extract hour from filename
def extract_hour(path):
    name = os.path.basename(path)
    parts = name.split("_")
    if len(parts) < 3:
        return None
    hh = parts[2].replace(".pt", "")[:2]
    return hh if hh.isdigit() else None

# collect prediction files
all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

# filter by target hour
filtered = [p for p in all_files if extract_hour(p) == target_hour]
print(f"Found {len(filtered)} files at hour={target_hour} UTC")

# main loop over files
for file_path in tqdm(filtered, desc="Accumulating pixelwise BSS"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception:
        continue

    gt   = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    pred = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))
    nfl  = np.nan_to_num(data["nflics"].astype(np.float32))

    if gt.shape != (H, W):
        continue

    gt   = np.clip(gt,   0, 1)
    pred = np.clip(pred, 0, 1)
    nfl  = np.clip(nfl,  0, 1)

    # enforce at least 15 convective cores in the raw scene
    gt_raw_bin = gt > 0.5
    if count_cores(gt_raw_bin) < 0:
        continue

    # apply neighbourhood max filter to ground truth
    gt_smooth = nd.maximum_filter(gt, size=L_pixels).astype(np.float32)

    # compute squared errors
    bs_m = (pred - gt_smooth)**2
    bs_r = (nfl  - gt_smooth)**2

    # accumulate
    bs_model_sum += bs_m
    bs_ref_sum   += bs_r
    count        += 1

# compute mean Brier scores
mask = count > 0
bs_model_mean = np.zeros((H, W), dtype=np.float32)
bs_ref_mean   = np.zeros((H, W), dtype=np.float32)

bs_model_mean[mask] = bs_model_sum[mask] / count[mask]
bs_ref_mean[mask]   = bs_ref_sum[mask]   / count[mask]

# compute pixelwise BSS
bss_map = np.full((H, W), np.nan, dtype=np.float32)
valid = (bs_ref_mean > 1e-12) & mask
bss_map[valid] = 1 - bs_model_mean[valid] / bs_ref_mean[valid]

# save output
out_file = os.path.join(
    output_dir,
    f"bss_pixelwise_hour_{target_hour}_t{lead_time}.npy"
)

np.save(out_file, bss_map)

print(f"Saved pixelwise BSS map (ground truth max-filtered, ≥15 cores) to {out_file}")
