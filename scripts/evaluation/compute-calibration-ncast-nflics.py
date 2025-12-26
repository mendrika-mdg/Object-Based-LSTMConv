import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.ndimage as nd
from scipy.ndimage import maximum_filter

lead_time = sys.argv[1]

base_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/predictions/ncast-nflics-full-corrected/t{lead_time}"

GT_FILTER_SIZE = 8
min_cores = 15

nbins = 20
bins = np.linspace(0, 1, nbins + 1)
bin_centres = 0.5 * (bins[:-1] + bins[1:])

def count_cores(mask):
    labelled, n = nd.label(mask > 0.5)
    return n

all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

print(f"Found {len(all_files)} files (all hours)")

counts_ncast = np.zeros(nbins, dtype=np.int64)
positives_ncast = np.zeros(nbins, dtype=np.int64)

counts_nflics = np.zeros(nbins, dtype=np.int64)
positives_nflics = np.zeros(nbins, dtype=np.int64)

for fp in tqdm(all_files, desc="Computing reliability"):
    try:
        data = torch.load(fp, weights_only=False)
    except Exception:
        continue

    gt_raw = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    ncast = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))
    nflics = np.nan_to_num(data["nflics"].astype(np.float32))

    gt_bin = (gt_raw > 0).astype(np.float32)

    if count_cores(gt_bin) < min_cores:
        continue

    gt_f = maximum_filter(gt_bin, size=GT_FILTER_SIZE)
    gt_flat = gt_f.reshape(-1)

    for pred, counts, positives in [
        (ncast, counts_ncast, positives_ncast),
        (nflics, counts_nflics, positives_nflics),
    ]:
        pr_flat = pred.reshape(-1)
        dig = np.digitize(pr_flat, bins) - 1
        dig = np.clip(dig, 0, nbins - 1)

        for i in range(nbins):
            m = dig == i
            if m.any():
                counts[i] += m.sum()
                positives[i] += gt_flat[m].sum()

obs_ncast = np.full(nbins, np.nan)
obs_nflics = np.full(nbins, np.nan)

for i in range(nbins):
    if counts_ncast[i] > 0:
        obs_ncast[i] = positives_ncast[i] / counts_ncast[i]
    if counts_nflics[i] > 0:
        obs_nflics[i] = positives_nflics[i] / counts_nflics[i]

df = pd.DataFrame({
    "bin_left": bins[:-1],
    "bin_right": bins[1:],
    "bin_centre": bin_centres,
    "obs_ncast": obs_ncast,
    "obs_nflics": obs_nflics,
    "counts_ncast": counts_ncast,
    "counts_nflics": counts_nflics,
})

out_csv = (
    f"/work/scratch-nopw2/mendrika/OB/evaluation/"
    f"ncast-nflics-full-corrected/reliability-full/"
    f"rel_allhours_t{lead_time}.csv"
)

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False)

print("Saved reliability to:", out_csv)
print("Total samples NCAST:", counts_ncast.sum())
print("Total samples NFLICS:", counts_nflics.sum())
