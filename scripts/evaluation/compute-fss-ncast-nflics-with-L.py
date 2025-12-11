import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import uniform_filter, maximum_filter
import scipy.ndimage as nd


lead_time = sys.argv[1]
target_hour = sys.argv[2]
base_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/predictions/ncast-nflics-full-corrected/t{lead_time}"

PIXEL_SIZE_KM = 3      # actual MSG output grid
GT_FILTER_SIZE = 8       # 25 km max filter → 25 / 3.1 ≈ 8 pixels
windows = [3, 9, 25, 49, 81, 121]   # FSS windows

def count_cores(mask):
    labelled, n = nd.label(mask > 0.5)
    return n

def compute_fss(pred, obs, window):
    pred = np.clip(pred, 0, 1)
    obs = np.clip(obs, 0, 1)
    f_pred = uniform_filter(pred, size=window, mode="constant")
    f_obs = uniform_filter(obs, size=window, mode="constant")
    num = np.mean((f_pred - f_obs) ** 2)
    den = np.mean(f_pred ** 2 + f_obs ** 2)
    return 1 - num / (den + 1e-8)


all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

def extract_hour(path):
    name = os.path.basename(path)
    timepart = name.split("_")[2].replace(".pt", "")
    return timepart[:2] if len(timepart) >= 2 else None

filtered_files = [p for p in all_files if extract_hour(p) == target_hour]
print(f"Found {len(filtered_files)} files at hour={target_hour} UTC")

fss_raw = {w: {"model": [], "persistence": [], "nflics": []} for w in windows}
fss_smooth = {w: {"model": [], "persistence": [], "nflics": []} for w in windows}

for file_path in tqdm(filtered_files, desc="Computing FSS"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        continue

    gt_raw = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    model = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))
    persistence_raw = np.nan_to_num(data["gt0"].cpu().numpy().astype(np.float32))
    nflics = np.nan_to_num(data["nflics"].astype(np.float32))

    # threshold raw GT
    gt_raw_bin = (gt_raw > 0).astype(np.float32)

    # skip scenes with few cores
    if count_cores(gt_raw_bin) < 15:
        continue

    # smoothed GT and persistence
    gt_smooth = maximum_filter(gt_raw_bin, size=GT_FILTER_SIZE)
    pers_smooth = maximum_filter(persistence_raw, size=GT_FILTER_SIZE)


    for w in windows:
        # RAW GT
        fss_raw[w]["model"].append(compute_fss(model, gt_raw_bin, w))
        fss_raw[w]["persistence"].append(compute_fss(persistence_raw, gt_raw_bin, w))
        fss_raw[w]["nflics"].append(compute_fss(nflics, gt_raw_bin, w))

        # SMOOTH GT
        fss_smooth[w]["model"].append(compute_fss(model, gt_smooth, w))
        fss_smooth[w]["persistence"].append(compute_fss(pers_smooth, gt_smooth, w))
        fss_smooth[w]["nflics"].append(compute_fss(nflics, gt_smooth, w))

rows = []

for w in windows:
    nominal_km = w * PIXEL_SIZE_KM

    r_mod = np.nanmean(fss_raw[w]["model"])
    r_per = np.nanmean(fss_raw[w]["persistence"])
    r_nfl = np.nanmean(fss_raw[w]["nflics"])

    s_mod = np.nanmean(fss_smooth[w]["model"])
    s_per = np.nanmean(fss_smooth[w]["persistence"])
    s_nfl = np.nanmean(fss_smooth[w]["nflics"])

    print(f"{w:>8} | {nominal_km:>12.1f} | {r_mod:>10.4f} | {r_per:>10.4f} | {r_nfl:>12.4f} | "
          f"{s_mod:>10.4f} | {s_per:>10.4f} | {s_nfl:>12.4f}")

    rows.append({
        "window_px": w,
        "nominal_scale_km": nominal_km,
        "raw_model": r_mod,
        "raw_persistence": r_per,
        "raw_nflics": r_nfl,
        "smooth_model": s_mod,
        "smooth_persistence": s_per,
        "smooth_nflics": s_nfl,
    })

# save CSV
output_csv = f"/work/scratch-nopw2/mendrika/OB/evaluation/ncast-nflics-full-corrected/fss/fss_hour_{target_hour}_t{lead_time}.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
pd.DataFrame(rows).to_csv(output_csv, index=False)

print(f"\nSaved combined RAW + SMOOTH FSS to {output_csv}")
