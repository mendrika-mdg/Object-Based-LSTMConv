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
# Hour extraction
# --------------------------------------------------
def extract_hour(path):
    name = os.path.basename(path)
    parts = name.split("_")
    if len(parts) < 3:
        return None
    timepart = parts[2].replace(".pt", "")
    return int(timepart[:2])

# --------------------------------------------------
# Hour bins (3-hour windows)
# --------------------------------------------------
bins_3h = {
    "00-03": range(0,  3),
    "03-06": range(3,  6),
    "06-09": range(6,  9),
    "09-12": range(9, 12),
    "12-15": range(12, 15),
    "15-18": range(15, 18),
    "18-21": range(18, 21),
    "21-24": range(21, 24),
}

# --------------------------------------------------
# Count convective cores
# --------------------------------------------------
def count_cores(mask):
    labelled, n = nd.label(mask > 0.5)
    return n

# --------------------------------------------------
# Gather all files
# --------------------------------------------------
all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            full_path = os.path.join(root, f)
            hour = extract_hour(full_path)
            if hour is not None:
                all_files.append((full_path, hour))

print(f"Found {len(all_files)} total files for lead time t+{lead_time}")

# --------------------------------------------------
# Reliability bin edges
# --------------------------------------------------
nbins = 20
bins = np.linspace(0, 1, nbins + 1)
bin_centres = 0.5 * (bins[:-1] + bins[1:])

# --------------------------------------------------
# Prepare storage per window
# --------------------------------------------------
window_counts = {w: np.zeros(nbins, dtype=np.int64) for w in bins_3h}
window_positives = {w: np.zeros(nbins, dtype=np.int64) for w in bins_3h}

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
for fp, hour in tqdm(all_files, desc="Processing files"):

    try:
        data = torch.load(fp, weights_only=False)
    except Exception as e:
        print("Skipping:", fp, e)
        continue

    # find window label
    window_label = None
    for label, hr_range in bins_3h.items():
        if hour in hr_range:
            window_label = label
            break
    if window_label is None:
        continue

    # arrays
    gt = np.asarray(data["gt"].cpu(), dtype=float)
    pred = np.asarray(data["mean"].cpu(), dtype=float)

    # MCS filter
    if count_cores(gt) < min_cores:
        continue

    # flatten
    gt_flat = gt.reshape(-1)
    pr_flat = pred.reshape(-1)

    # assign bins
    digitised = np.digitize(pr_flat, bins) - 1
    digitised = np.clip(digitised, 0, nbins - 1)

    # update window aggregates
    wc = window_counts[window_label]
    wp = window_positives[window_label]

    for i in range(nbins):
        m = digitised == i
        if m.any():
            wc[i] += m.sum()
            wp[i] += gt_flat[m].sum()

# --------------------------------------------------
# Save each window separately
# --------------------------------------------------
out_dir = "/work/scratch-nopw2/mendrika/OB/evaluation/ncast-nflics/reliability/timewindows"
os.makedirs(out_dir, exist_ok=True)

for label in bins_3h.keys():

    counts = window_counts[label]
    positives = window_positives[label]

    obs_freq = np.zeros(nbins)
    for i in range(nbins):
        if counts[i] > 0:
            obs_freq[i] = positives[i] / counts[i]
        else:
            obs_freq[i] = np.nan

    df = pd.DataFrame({
        "bin_left": bins[:-1],
        "bin_right": bins[1:],
        "bin_centre": bin_centres,
        "obs_freq": obs_freq,
        "counts": counts,
    })

    out_csv = os.path.join(out_dir, f"rel_t{lead_time}_{label}.csv")
    df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}  (samples={counts.sum()})")

print("\nAll windows finished.")
