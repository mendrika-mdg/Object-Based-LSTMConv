import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import scipy.ndimage as nd

# arguments
lead_time = sys.argv[1]
target_hour = sys.argv[2]

# directories
base_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/predictions/ncast-calibrated/t{lead_time}"
output_dir = "/home/users/mendrika/Object-Based-LSTMConv/outputs/evaluation/ncast-calibrated/bss"
os.makedirs(output_dir, exist_ok=True)

# expected map size
map_shape = (512, 512)

# count cores
def count_cores(mask):
    labelled, n = nd.label(mask > 0.5)
    return n

# normal BSS
def compute_bss_normal(pred_model, pred_ref, obs):
    pred_model = np.clip(pred_model, 0, 1)
    pred_ref   = np.clip(pred_ref, 0, 1)
    obs        = np.clip(obs, 0, 1)
    bs_model = np.mean((pred_model - obs)**2)
    bs_ref   = np.mean((pred_ref   - obs)**2)
    if bs_ref < 1e-12:
        return np.nan
    return 1 - bs_model / (bs_ref + 1e-12)

# restricted BSS
def compute_bss_restricted(pred_model, pred_ref, obs):
    pred_model = np.clip(pred_model, 0, 1)
    pred_ref   = np.clip(pred_ref, 0, 1)
    obs        = np.clip(obs, 0, 1)
    bs_model = (pred_model - obs)**2
    bs_ref   = (pred_ref   - obs)**2
    valid = bs_ref > 1e-12
    conv = (obs > 0) | (pred_model > 0.1) | (pred_ref > 0)
    valid = valid & conv
    if np.sum(valid) < 10:
        return np.nan
    return 1 - np.mean(bs_model[valid]) / (np.mean(bs_ref[valid]) + 1e-12)

# recursive file search
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
    hh = parts[2].replace(".pt", "")[:2]
    return hh if hh.isdigit() else None

# filter by desired hour
filtered_files = [p for p in all_files if extract_hour(p) == target_hour]
print(f"Found {len(filtered_files)} files at hour={target_hour} UTC")

# storage
normal_bss_model = []
rest_bss_model   = []

# main loop
for file_path in tqdm(filtered_files, desc="Computing BSS"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception:
        continue

    gt       = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    pred_cal = np.nan_to_num(data["mean_cal"].cpu().numpy().astype(np.float32))
    pred_raw = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))
    persist  = np.nan_to_num(data["gt0"].cpu().numpy().astype(np.float32))

    ncores = count_cores(gt)
    if ncores < 15:
        continue

    for arr in [gt, pred_cal, persist]:
        if arr.shape != map_shape:
            raise ValueError("Shape mismatch")

    pred_model = pred_cal
    # pred_model = pred_raw

    n_model = compute_bss_normal(pred_model, persist, gt)
    r_model = compute_bss_restricted(pred_model, persist, gt)

    normal_bss_model.append(n_model)
    rest_bss_model.append(r_model)

# averages
mean_normal_model = float(np.nanmean(normal_bss_model))
mean_rest_model   = float(np.nanmean(rest_bss_model))

# save
np.save(os.path.join(output_dir, f"bss_normal_model_hour_{target_hour}_t{lead_time}.npy"),
        mean_normal_model)

np.save(os.path.join(output_dir, f"bss_restricted_model_hour_{target_hour}_t{lead_time}.npy"),
        mean_rest_model)

print("Normal BSS (model):     ", mean_normal_model)
print("Restricted BSS (model): ", mean_rest_model)
