import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import uniform_filter
import scipy.ndimage as nd

# arguments
lead_time = sys.argv[1]
target_hour = sys.argv[2]

# directories
base_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/predictions/ncast-calibrated/t{lead_time}"

# windows
windows = [3, 9, 25, 49, 81, 121]
PIXEL_SIZE_KM = 3

# count convective cores
def count_cores(mask):
    labelled, n = nd.label(mask > 0.5)
    return n

# fss function
def compute_fss(pred, obs, window):
    pred = np.clip(pred, 0, 1)
    obs = np.clip(obs, 0, 1)
    f_pred = uniform_filter(pred, size=window, mode="constant")
    f_obs = uniform_filter(obs, size=window, mode="constant")
    num = np.mean((f_pred - f_obs) ** 2)
    den = np.mean(f_pred ** 2 + f_obs ** 2)
    return 1 - num / (den + 1e-8)

# recursive search
all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

# extract hour from filename
def extract_hour(path):
    name = os.path.basename(path)
    parts = name.split("_")
    if len(parts) < 3:
        return None
    timepart = parts[2].replace(".pt", "")
    if len(timepart) < 2:
        return None
    return timepart[:2]

# filter files by hour
filtered_files = [p for p in all_files if extract_hour(p) == target_hour]
print(f"Found {len(filtered_files)} files at hour={target_hour} UTC")

# storage
fss_results = {
    w: {"raw": [], "cal": [], "persistence": []}
    for w in windows
}

# main loop
for file_path in tqdm(filtered_files, desc="Computing FSS"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        continue

    gt = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    pred_cal = np.nan_to_num(data["mean_cal"].cpu().numpy().astype(np.float32))
    pred_raw = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))
    persistence = np.nan_to_num(data["gt0"].cpu().numpy().astype(np.float32))

    ncores = count_cores(gt)
    if ncores < 15:
        continue

    for w in windows:
        fss_results[w]["raw"].append(compute_fss(pred_raw, gt, w))
        fss_results[w]["cal"].append(compute_fss(pred_cal, gt, w))
        fss_results[w]["persistence"].append(compute_fss(persistence, gt, w))

# summary
rows = []
print(f"\nAverage FSS for hour={target_hour} UTC (lead t+{lead_time}), filtered by ≥15 cores")
print(f"{'Window':>8} | {'Scale(km)':>10} | {'Raw':>10} | {'Cal':>10} | {'Persist':>10}")
print("-" * 70)

for w in windows:
    scale = w * PIXEL_SIZE_KM
    mean_raw = np.nanmean(fss_results[w]["raw"])
    mean_cal = np.nanmean(fss_results[w]["cal"])
    mean_persistence = np.nanmean(fss_results[w]["persistence"])

    rows.append({
        "window": w,
        "scale_km": scale,
        "raw": mean_raw,
        "cal": mean_cal,
        "persistence": mean_persistence,
    })

    print(f"{w:>8} | {scale:>10.0f} | {mean_raw:.4f} | {mean_cal:.4f} | {mean_persistence:.4f}")

# save
output_csv = f"/work/scratch-nopw2/mendrika/OB/evaluation/ncast-calibrated/fss/fss_hour_{target_hour}_t{lead_time}.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
pd.DataFrame(rows).to_csv(output_csv, index=False)

print(f"\nSaved FSS summary to {output_csv}")
