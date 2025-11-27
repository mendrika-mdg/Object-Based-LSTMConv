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
output_dir = "/home/users/mendrika/Object-Based-LSTMConv/outputs/evaluation/ncast/bss/improved"
os.makedirs(output_dir, exist_ok=True)

# map size
map_shape = (512, 512)

# restricted BSS
def compute_bss_restricted(pred_model, pred_ref, obs):
    pred_model = np.clip(pred_model, 0, 1)
    pred_ref   = np.clip(pred_ref, 0, 1)
    obs        = np.clip(obs, 0, 1)

    bs_model = (pred_model - obs)**2
    bs_ref   = (pred_ref   - obs)**2

    # mask out trivial pixels where persistence == obs == 0
    valid = bs_ref > 1e-12

    # restrict to convective neighbourhood
    convective_mask = (obs > 0) | (pred_model > 0.1) | (pred_ref > 0)
    valid = valid & convective_mask

    if np.sum(valid) < 10:
        return np.nan

    bs_model_mean = np.mean(bs_model[valid])
    bs_ref_mean   = np.mean(bs_ref[valid])

    return 1 - bs_model_mean / (bs_ref_mean + 1e-12)


# recursive search
all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

# extract hour from filename
def extract_hour(path):
    name = os.path.basename(path)
    parts = name.split("_")
    if len(parts) < 3:
        return None
    timepart = parts[2].replace(".pt", "")
    if len(timepart) < 2:
        return None
    return timepart[:2]

# filter by target hour
filtered_files = [p for p in all_files if extract_hour(p) == target_hour]
print(f"Found {len(filtered_files)} files at hour={target_hour} UTC")

# store BSS values
bss_model_list = []
bss_persist_list = []

# loop through files
for file_path in tqdm(filtered_files, desc="Computing restricted BSS"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        continue

    gt       = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    model    = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))
    persist  = np.nan_to_num(data["gt0"].cpu().numpy().astype(np.float32))

    for name, arr in zip(["gt", "model", "persistence"], [gt, model, persist]):
        if arr.shape != map_shape:
            raise ValueError(f"{name} has shape {arr.shape}, expected {map_shape}")

    # restricted BSS (model vs persistence)
    bss_model   = compute_bss_restricted(model,   persist, gt)
    bss_persist = compute_bss_restricted(persist, persist, gt)

    bss_model_list.append(bss_model)
    bss_persist_list.append(bss_persist)

# mean across all valid samples
mean_bss_model       = float(np.nanmean(bss_model_list))
mean_bss_persistence = float(np.nanmean(bss_persist_list))

# save arrays
np.save(os.path.join(output_dir, f"bss_model_hour_{target_hour}_t{lead_time}.npy"),
        mean_bss_model)
np.save(os.path.join(output_dir, f"bss_persistence_hour_{target_hour}_t{lead_time}.npy"),
        mean_bss_persistence)

print("Saved restricted BSS:")
print("Model BSS:      ", mean_bss_model)
print("Persistence BSS:", mean_bss_persistence)
