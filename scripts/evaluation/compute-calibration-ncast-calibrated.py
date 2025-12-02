import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.ndimage as nd
from tqdm import tqdm

# arguments
lead_time = sys.argv[1]
min_cores = 15

base_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/predictions/ncast-calibrated/t{lead_time}"

# count convective cores
def count_cores(mask):
    labelled, n = nd.label(mask > 0.5)
    return n

# find all prediction files
all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

print(f"Found {len(all_files)} files for lead time t+{lead_time}")

# reliability bins
nbins = 20
bins = np.linspace(0, 1, nbins + 1)
bin_centres = 0.5 * (bins[:-1] + bins[1:])

# running aggregates for raw
counts_raw = np.zeros(nbins, dtype=np.int64)
positives_raw = np.zeros(nbins, dtype=np.int64)

# running aggregates for calibrated
counts_cal = np.zeros(nbins, dtype=np.int64)
positives_cal = np.zeros(nbins, dtype=np.int64)

# main loop
for fp in tqdm(all_files, desc="Computing reliability"):

    try:
        data = torch.load(fp, weights_only=False)
    except Exception as e:
        print("Skipping:", fp, e)
        continue

    gt = np.asarray(data["gt"].cpu(), dtype=float)
    pred_raw = np.asarray(data["mean"].cpu(), dtype=float)
    pred_cal = np.asarray(data["mean_cal"].cpu(), dtype=float)

    # MCS filter
    if count_cores(gt) < min_cores:
        continue

    # flatten
    gt_flat = gt.reshape(-1)
    pr_raw = pred_raw.reshape(-1)
    pr_cal = pred_cal.reshape(-1)

    # digitise raw
    dig_raw = np.digitize(pr_raw, bins) - 1
    dig_raw = np.clip(dig_raw, 0, nbins - 1)

    # digitise calibrated
    dig_cal = np.digitize(pr_cal, bins) - 1
    dig_cal = np.clip(dig_cal, 0, nbins - 1)

    # update both sets
    for i in range(nbins):

        m_raw = dig_raw == i
        if m_raw.any():
            counts_raw[i] += m_raw.sum()
            positives_raw[i] += gt_flat[m_raw].sum()

        m_cal = dig_cal == i
        if m_cal.any():
            counts_cal[i] += m_cal.sum()
            positives_cal[i] += gt_flat[m_cal].sum()

# observed frequencies raw
obs_raw = np.zeros(nbins)
for i in range(nbins):
    if counts_raw[i] > 0:
        obs_raw[i] = positives_raw[i] / counts_raw[i]
    else:
        obs_raw[i] = np.nan

# observed frequencies calibrated
obs_cal = np.zeros(nbins)
for i in range(nbins):
    if counts_cal[i] > 0:
        obs_cal[i] = positives_cal[i] / counts_cal[i]
    else:
        obs_cal[i] = np.nan

# save CSV
df = pd.DataFrame({
    "bin_left": bins[:-1],
    "bin_right": bins[1:],
    "bin_centre": bin_centres,
    "obs_raw": obs_raw,
    "obs_cal": obs_cal,
    "counts_raw": counts_raw,
    "counts_cal": counts_cal
})

out_csv = f"/work/scratch-nopw2/mendrika/OB/evaluation/ncast-calibrated/reliability/rel_t{lead_time}.csv"
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False)

print("\nSaved reliability to:", out_csv)
print("Total raw samples:", counts_raw.sum())
print("Total calibrated samples:", counts_cal.sum())
