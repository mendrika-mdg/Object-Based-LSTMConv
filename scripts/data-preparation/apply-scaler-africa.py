import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

# check args
if len(sys.argv) != 2:
    print("Usage: python apply_scaler_instances.py <partition>")
    sys.exit(1)

PARTITION = sys.argv[1]

# paths
SCALER_PATH = "/home/users/mendrika/Object-Based-LSTMConv/outputs/scaler-africa/scaler_realcores_online.pt"
SPLIT_JSON  = "/home/users/mendrika/Object-Based-LSTMConv/outputs/data-split/input_splits.json"

SAVE_BASE = "/work/scratch-nopw2/mendrika/pancast/preprocessed/inputs_t0"
os.makedirs(SAVE_BASE, exist_ok=True)

# feature layout
# [month_sin, month_cos, tod_sin, tod_cos,
#  lat, lon, lat_min, lat_max, lon_min, lon_max,
#  tir, size, mask]

MASK_COL_INDEX = 12
COLS_TO_SCALE = list(range(4, 12))

# load scaler
try:
    scaler = torch.load(SCALER_PATH, map_location="cpu", weights_only=False)
except Exception as e:
    print(f"Failed to load scaler: {e}")
    sys.exit(1)

mean = torch.tensor(scaler["mean"], dtype=torch.float32)
scale = torch.tensor(scaler["scale"], dtype=torch.float32)

print(f"Loaded scaler from {SCALER_PATH}")
print(f"Applying scaler to partition={PARTITION.upper()}")

# load split
try:
    with open(SPLIT_JSON, "r") as f:
        splits = json.load(f)
except Exception as e:
    print(f"Failed to load split file: {e}")
    sys.exit(1)

if PARTITION not in splits:
    print(f"Partition '{PARTITION}' not found in split file")
    sys.exit(1)

files = splits[PARTITION]
print(f"Found {len(files):,} files to process")

def apply_scaler_instance(x):
    T, N, F = x.shape
    flat = x.view(T * N, F)
    flat[:, COLS_TO_SCALE] = (flat[:, COLS_TO_SCALE] - mean) / scale
    return flat.view(T, N, F)

out_dir = os.path.join(SAVE_BASE, PARTITION)
os.makedirs(out_dir, exist_ok=True)

n_ok = 0
n_failed = 0

for fpath in tqdm(files, desc=f"Scaling {PARTITION}"):

    try:
        data = torch.load(fpath, map_location="cpu")
    except Exception as e:
        print(f"Failed to load {fpath}: {e}")
        n_failed += 1
        continue

    if "input_tensor" not in data:
        print(f"Missing input_tensor in {fpath}")
        n_failed += 1
        continue

    x = data["input_tensor"]

    if not isinstance(x, torch.Tensor) or x.ndim != 3:
        print(f"Invalid input_tensor in {fpath}")
        n_failed += 1
        continue

    try:
        x_scaled = apply_scaler_instance(x)
    except Exception as e:
        print(f"Scaling failed for {fpath}: {e}")
        n_failed += 1
        continue

    out = {
        "input_tensor": x_scaled,
        "nowcast_origin": data.get("nowcast_origin", "unknown"),
    }

    out_path = os.path.join(out_dir, os.path.basename(fpath))

    try:
        torch.save(out, out_path)
        n_ok += 1
    except Exception as e:
        print(f"Failed to save {out_path}: {e}")
        n_failed += 1

print(f"Finished scaling {PARTITION}")
print(f"Successful files : {n_ok}")
print(f"Failed files     : {n_failed}")
print(f"Saved to {out_dir}")
