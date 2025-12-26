import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Args
PARTITION = sys.argv[1]               # train / val / test
LEAD_TIME = int(sys.argv[2])          # minutes

# Paths
BASE_DIR = "/work/scratch-nopw2/mendrika/pancast/raw"
RAW_INPUT_DIR = f"{BASE_DIR}/inputs_t0"
RAW_TARGET_DIR = f"{BASE_DIR}/targets_t{LEAD_TIME:03d}min"
SPLIT_FILE = f"/home/users/mendrika/Object-Based-LSTMConv/outputs/data-split/{PARTITION}_files.txt"
SHARDS_DIR = f"{BASE_DIR}/shards/t{LEAD_TIME:03d}min/{PARTITION}"

os.makedirs(SHARDS_DIR, exist_ok=True)

FILES_PER_SHARD = 1000

print(f"Creating shards | partition={PARTITION.upper()} | lead={LEAD_TIME} min")
print(f"Input dir : {RAW_INPUT_DIR}")
print(f"Target dir: {RAW_TARGET_DIR}")
print(f"Shard dir : {SHARDS_DIR}")

# Load file list
with open(SPLIT_FILE) as f:
    files = [line.strip() for line in f if line.strip()]

print(f"Total files listed: {len(files):,}")

# Buffers
X_buf, Y_buf, ID_buf = [], [], []
shard_idx = 0
n_saved = 0
n_skipped = 0

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

# Main loop
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

        # Load target
        fname = os.path.basename(fpath).replace("input-", "target-")
        target_path = os.path.join(RAW_TARGET_DIR, fname)

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

        # Convert to NumPy
        X_buf.append(x.cpu().numpy().astype(np.float32))
        Y_buf.append(y.cpu().numpy().astype(np.uint8))
        ID_buf.append(nowcast_id)

        if len(X_buf) == FILES_PER_SHARD:
            flush_shard()

    except Exception as e:
        print(f"Error processing {fpath}: {e}")
        n_skipped += 1

# Flush remainder
flush_shard()

print(f"Finished sharding {PARTITION.upper()} | LT={LEAD_TIME} min")
print(f"Total saved samples : {n_saved:,}")
print(f"Total skipped       : {n_skipped:,}")
print(f"Total shards        : {shard_idx}")
