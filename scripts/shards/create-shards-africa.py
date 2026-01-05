import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

PARTITION = sys.argv[1]        # train / val / test
LEAD_TIME = int(sys.argv[2])   # minutes

SPLIT_JSON = "/home/users/mendrika/Object-Based-LSTMConv/outputs/data-split/input_splits.json"
SCALER_PATH = "/home/users/mendrika/Object-Based-LSTMConv/outputs/scaler-africa/scaler_realcores_online.pt"

SHARDS_DIR = f"/gws/ssde/j25b/swift/mendrika/pancast/shards/t{LEAD_TIME:03d}min/{PARTITION}"
os.makedirs(SHARDS_DIR, exist_ok=True)

FILES_PER_SHARD = 200

print(f"Creating shards | partition={PARTITION.upper()} | lead={LEAD_TIME} min")
print(f"Shard dir : {SHARDS_DIR}")

# feature layout
# [month_sin, month_cos, tod_sin, tod_cos,
#  lat, lon, lat_min, lat_max, lon_min, lon_max,
#  tir, size, mask]

COLS_TO_SCALE = list(range(4, 12))

# load scaler (fitted on real cores only)
scaler = torch.load(SCALER_PATH, map_location="cpu", weights_only=False)
mean = torch.tensor(scaler["mean"], dtype=torch.float32)
scale = torch.tensor(scaler["scale"], dtype=torch.float32)

# Load split lists
with open(SPLIT_JSON, "r") as f:
    splits = json.load(f)

files = splits[PARTITION]
print(f"Total files listed: {len(files):,}")

X_buf, Y_buf, ID_buf = [], [], []
shard_idx = 0
n_saved = 0
n_skipped = 0

def apply_scaler(x):
    T, N, F = x.shape
    flat = x.view(T * N, F)
    flat[:, COLS_TO_SCALE] = (flat[:, COLS_TO_SCALE] - mean) / scale
    return flat.view(T, N, F)

def flush_shard():
    global shard_idx, X_buf, Y_buf, ID_buf, n_saved

    if len(X_buf) == 0:
        return

    shard_path = os.path.join(SHARDS_DIR, f"shard_{shard_idx:04d}.pt")

    torch.save(
        {
            "X": torch.from_numpy(np.stack(X_buf)).float(),
            "Y": torch.from_numpy(np.stack(Y_buf)).to(torch.uint8),
            "ID": ID_buf,
        },
        shard_path,
    )

    print(f"Saved shard_{shard_idx:04d} | {len(X_buf)} samples")
    n_saved += len(X_buf)
    shard_idx += 1
    X_buf, Y_buf, ID_buf = [], [], []

for fpath in tqdm(files, desc="Sharding"):
    try:
        data_in = torch.load(fpath, map_location="cpu")

        if "input_tensor" not in data_in:
            n_skipped += 1
            continue

        x = data_in["input_tensor"]
        if x.ndim != 3:
            n_skipped += 1
            continue

        nowcast_id = data_in.get("nowcast_origin", os.path.basename(fpath))

        parent_dir = os.path.dirname(os.path.dirname(fpath))
        target_dir = os.path.join(parent_dir, f"targets_t{LEAD_TIME:03d}min")
        target_name = os.path.basename(fpath).replace("input-", "target-")
        target_path = os.path.join(target_dir, target_name)

        if not os.path.exists(target_path):
            n_skipped += 1
            continue

        data_out = torch.load(target_path, map_location="cpu")
        y = data_out.get("data", None)

        if y is None:
            n_skipped += 1
            continue

        if y.ndim != 2:
            y = y.squeeze()

        # apply scaling here
        x = apply_scaler(x)

        X_buf.append(x.cpu().numpy().astype(np.float32))
        Y_buf.append(y.cpu().numpy().astype(np.uint8))
        ID_buf.append(nowcast_id)

        if len(X_buf) == FILES_PER_SHARD:
            flush_shard()

    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        n_skipped += 1

flush_shard()

print(f"Finished sharding {PARTITION.upper()} | LT={LEAD_TIME} min")
print(f"Total saved samples : {n_saved:,}")
print(f"Total skipped       : {n_skipped:,}")
print(f"Total shards        : {shard_idx}")
