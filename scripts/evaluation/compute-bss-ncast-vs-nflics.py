import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import scipy.ndimage as nd

# arguments
lead_time = sys.argv[1]
target_hour = sys.argv[2]

# directories
base_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/predictions/ncast-nflics-full/t{lead_time}"
output_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/ncast-nflics-full/bss"
os.makedirs(output_dir, exist_ok=True)

H, W = 512, 512

# 25 km → 9 px max filter
L_pixels = 9

# accumulators
bs_model_sum = np.zeros((H, W), dtype=np.float64)
bs_ref_sum   = np.zeros((H, W), dtype=np.float64)
count        = np.zeros((H, W), dtype=np.int32)

def extract_hour(path):
    name = os.path.basename(path)
    parts = name.split("_")
    if len(parts) < 3:
        return None
    hh = parts[2].replace(".pt", "")[:2]
    return hh if hh.isdigit() else None

# collect files
all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

filtered = [p for p in all_files if extract_hour(p) == target_hour]
print(f"Found {len(filtered)} files at hour={target_hour} UTC")

# main loop
for file_path in tqdm(filtered, desc="Accumulating pixelwise BSS"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception:
        continue

    gt     = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    pred   = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))
    nfl    = np.nan_to_num(data["nflics"].astype(np.float32))

    if gt.shape != (H, W):
        continue

    gt   = np.clip(gt,   0, 1)
    pred = np.clip(pred, 0, 1)
    nfl  = np.clip(nfl,  0, 1)

    # ---- apply 25 km neighbourhood smoothing to GT ----
    gt_smooth = nd.maximum_filter(gt, size=L_pixels).astype(np.float32)

    # ---- compute squared errors ----
    bs_m = (pred - gt_smooth)**2
    bs_r = (nfl  - gt_smooth)**2

    bs_model_sum += bs_m
    bs_ref_sum   += bs_r
    count        += 1

# compute per-pixel mean BS
mask = count > 0
bs_model_mean = np.zeros((H, W), dtype=np.float32)
bs_ref_mean   = np.zeros((H, W), dtype=np.float32)

bs_model_mean[mask] = bs_model_sum[mask] / count[mask]
bs_ref_mean[mask]   = bs_ref_sum[mask]   / count[mask]

# compute BSS map
bss_map = np.full((H, W), np.nan, dtype=np.float32)
valid = (bs_ref_mean > 1e-12) & mask
bss_map[valid] = 1 - bs_model_mean[valid] / bs_ref_mean[valid]

# save result
out_file = os.path.join(
    output_dir,
    f"bss_pixelwise_hour_{target_hour}_t{lead_time}.npy"
)

np.save(out_file, bss_map)

print(f"Saved pixelwise BSS map (GT max-filtered) to {out_file}")
