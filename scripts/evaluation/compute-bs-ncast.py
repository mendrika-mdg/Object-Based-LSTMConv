import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# arguments
lead_time = sys.argv[1]
target_hour = sys.argv[2]

# directories
base_dir = f"/work/scratch-nopw2/mendrika/OB/predictions/ncast/t{lead_time}"
output_dir = "/home/users/mendrika/Object-Based-LSTMConv/outputs/evaluation/ncast/bs"
os.makedirs(output_dir, exist_ok=True)

# map size
map_shape = (512, 512)

# bs function
def compute_bs(pred, obs):
    pred = np.clip(pred, 0, 1)
    obs  = np.clip(obs, 0, 1)
    return (pred - obs) ** 2

# recursive search
all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

# extract hour
def extract_hour(path):
    name = os.path.basename(path)
    parts = name.split("_")
    if len(parts) < 3:
        return None
    timepart = parts[2].replace(".pt", "")
    if len(timepart) < 2:
        return None
    return timepart[:2]

# filter by hour
filtered_files = [p for p in all_files if extract_hour(p) == target_hour]
print(f"Found {len(filtered_files)} files at hour={target_hour} UTC")

# storage
bs_model_list = []
bs_persist_list = []

# loop
for file_path in tqdm(filtered_files, desc="Computing BS"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        continue

    gt   = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    model = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))
    persist = np.nan_to_num(data["gt0"].cpu().numpy().astype(np.float32))

    for name, arr in zip(["gt", "model", "persistence"], [gt, model, persist]):
        if arr.shape != map_shape:
            raise ValueError(f"{name} has shape {arr.shape}, expected {map_shape}")

    bs_model = compute_bs(model, gt)
    bs_persist = compute_bs(persist, gt)

    bs_model_list.append(bs_model)
    bs_persist_list.append(bs_persist)

# stack and mean
mean_bs_model = np.nanmean(np.stack(bs_model_list, axis=0), axis=0)
mean_bs_persist = np.nanmean(np.stack(bs_persist_list, axis=0), axis=0)

# save
np.save(os.path.join(output_dir, f"bs_model_hour_{target_hour}_t{lead_time}.npy"), mean_bs_model)
np.save(os.path.join(output_dir, f"bs_persist_hour_{target_hour}_t{lead_time}.npy"), mean_bs_persist)
