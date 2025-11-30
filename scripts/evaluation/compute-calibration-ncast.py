import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.ndimage as nd
from tqdm import tqdm

# --------------------------------------------------
# Arguments
# --------------------------------------------------
lead_time = sys.argv[1]       # "1", "3", or "6"
min_cores = 15                # MCS filter

base_dir = f"/work/scratch-nopw2/mendrika/OB/predictions/ncast-nflics/t{lead_time}"

# --------------------------------------------------
# Count convective cores
# --------------------------------------------------
def count_cores(mask):
    labelled, n = nd.label(mask > 0.5)
    return n

# --------------------------------------------------
# Find all prediction files
# --------------------------------------------------
all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

print(f"Found {len(all_files)} files for lead time t+{lead_time}")

# --------------------------------------------------
# Reliability bins (20 bins)
# --------------------------------------------------
nbins = 20
bins = np.linspace(0, 1, nbins + 1)
bin_centres = 0.5 * (bins[:-1] + bins[1:])

# Running aggregates (constant memory)
counts = np.zeros(nbins, dtype=np.int64)
positives = np.zeros(nbins, dtype=np.int64)

# --------------------------------------------------
# MAIN LOOP — streaming accumulation
# --------------------------------------------------
for fp in tqdm(all_files, desc="Computing reliability"):

    try:
        data = torch.load(fp, weights_only=False)
    except Exception as e:
        print("Skipping:", fp, e)
        continue

    gt = np.asarray(data["gt"].cpu(), dtype=float)
    pred = np.asarray(data["mean"].cpu(), dtype=float)

    # MCS filtering
    if count_cores(gt) < min_cores:
        continue

    # flatten pixel arrays
    gt_flat = gt.reshape(-1)
    pr_flat = pred.reshape(-1)

    # assign each pixel to a bin
    digitised = np.digitize(pr_flat, bins) - 1
    digitised = np.clip(digitised, 0, nbins - 1)

    # update counts
    for i in range(nbins):
        m = digitised == i
        if m.any():
            counts[i] += m.sum()
            positives[i] += gt_flat[m].sum()

# --------------------------------------------------
# Compute observed frequencies
# --------------------------------------------------
obs_freq = np.zeros(nbins)
for i in range(nbins):
    if counts[i] > 0:
        obs_freq[i] = positives[i] / counts[i]
    else:
        obs_freq[i] = np.nan

# --------------------------------------------------
# Save to CSV
# --------------------------------------------------
df = pd.DataFrame({
    "bin_left": bins[:-1],
    "bin_right": bins[1:],
    "bin_centre": bin_centres,
    "obs_freq": obs_freq,
    "counts": counts
})

out_csv = f"/work/scratch-nopw2/mendrika/OB/evaluation/ncast-nflics/reliability/rel_t{lead_time}.csv"
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False)

print("\nSaved reliability to:", out_csv)
print("Total samples used:", counts.sum())
