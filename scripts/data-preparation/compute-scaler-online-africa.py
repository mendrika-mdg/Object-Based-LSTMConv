import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Paths
SPLIT_JSON = "/home/users/mendrika/Object-Based-LSTMConv/outputs/data-split/input_splits.json"
SAVE_DIR = "/home/users/mendrika/Object-Based-LSTMConv/outputs/scaler-africa"

os.makedirs(SAVE_DIR, exist_ok=True)

# Feature layout:
# [month_sin, month_cos, tod_sin, tod_cos,
#  lat, lon, lat_min, lat_max, lon_min, lon_max,
#  tir, size, mask]

MASK_COL_INDEX = 12

# Only scale continuous physical features
COLS_TO_SCALE = list(range(4, 12))

scaler = StandardScaler()

# Load split lists
with open(SPLIT_JSON, "r") as f:
    splits = json.load(f)

train_files = splits["train"]

print(f"Computing scaler from {len(train_files):,} training input files")

n_real_total = 0
n_files_used = 0
n_files_skipped = 0

for fpath in tqdm(train_files, desc="Processing inputs"):
    try:
        data = torch.load(fpath, map_location="cpu")

        if "input_tensor" not in data:
            n_files_skipped += 1
            continue

        x = data["input_tensor"]  # (T, Ncore, F)

        if x.ndim != 3:
            n_files_skipped += 1
            continue

        T, N, F = x.shape

        # Flatten to (T * Ncore, F)
        flat = x.view(T * N, F).cpu().numpy()

        # Select real (non-artificial) cores
        real = flat[flat[:, MASK_COL_INDEX] == 1]

        if real.shape[0] == 0:
            continue

        # Online update (float64 for numerical stability)
        scaler.partial_fit(real[:, COLS_TO_SCALE].astype(np.float64))

        n_real_total += real.shape[0]
        n_files_used += 1

    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        n_files_skipped += 1

print()
print(f"Files used        : {n_files_used:,}")
print(f"Files skipped     : {n_files_skipped:,}")
print(f"Real cores total  : {n_real_total:,}")

# Save scaler statistics
stats = {
    "mean": scaler.mean_,
    "scale": scaler.scale_,
    "var": scaler.var_,
    "cols": COLS_TO_SCALE,
    "mask_col": MASK_COL_INDEX,
    "n_real_cores": n_real_total,
}

out_path = os.path.join(SAVE_DIR, "scaler_realcores_online.pt")
torch.save(stats, out_path)

print(f"Scaler saved to {out_path}")
print("Done.")
