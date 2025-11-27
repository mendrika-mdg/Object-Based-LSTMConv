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
output_dir = "/home/users/mendrika/Object-Based-LSTMConv/outputs/evaluation/ncast/bss"
os.makedirs(output_dir, exist_ok=True)

# map size
map_shape = (512, 512)

# bss function
def compute_bss(pred_model, pred_ref, obs):
    pred_model = np.clip(pred_model, 0, 1)
    pred_ref   = np.clip(pred_ref, 0, 1)
    obs        = np.clip(obs, 0, 1)
    bs_model = (pred_model - obs) ** 2
    bs_ref   = (pred_ref - obs) ** 2
    return 1 - bs_model / (bs_ref + 1e-8)

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
bss_model_list = []
bss_persist_list = []

# loop
for file_path in tqdm(filtered_files, desc="Computing BSS"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        continue

    gt = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    model = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))
    persist = np.nan_to_num(data["gt0"].cpu().numpy().astype(np.float32))

    for name, arr in zip(["gt", "model", "persistence"], [gt, model, persist]):
        if arr.shape != map_shape:
            raise ValueError(f"{name} has shape {arr.shape}, expected {map_shape}")

    bss_model = compute_bss(model, persist, gt)
    bss_pers  = compute_bss(persist, persist, gt)

    bss_model_list.append(bss_model)
    bss_persist_list.append(bss_pers)

# stack and mean
mean_bss_model = np.nanmean(np.stack(bss_model_list, axis=0), axis=0)
mean_bss_persist = np.nanmean(np.stack(bss_persist_list, axis=0), axis=0)

# save
np.save(os.path.join(output_dir, f"bss_model_hour_{target_hour}_t{lead_time}.npy"), mean_bss_model)
np.save(os.path.join(output_dir, f"bss_persistence_hour_{target_hour}_t{lead_time}.npy"), mean_bss_persist)
