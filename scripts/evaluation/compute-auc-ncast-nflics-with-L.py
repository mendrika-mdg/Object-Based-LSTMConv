import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.ndimage import maximum_filter

# arguments
lead_time = sys.argv[1]
target_hour = sys.argv[2]

# directories
base_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/predictions/ncast-nflics-full/t{lead_time}"
output_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/ncast-nflics-full/auc"
os.makedirs(output_dir, exist_ok=True)

# map size
map_shape = (512, 512)

# neighbourhood size for GT and persistence
# NFLICS already has its own neighbourhood applied – DO NOT filter it again
L_pixels = 8

# recursive search for .pt files
all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

def extract_hour(path):
    name = os.path.basename(path)
    parts = name.split("_")
    if len(parts) < 3:
        return None
    tp = parts[2].replace(".pt", "")
    if len(tp) < 2:
        return None
    return tp[:2]

filtered_files = [p for p in all_files if extract_hour(p) == target_hour]
print(f"Found {len(filtered_files)} files at hour={target_hour} UTC")

# containers
all_gt = []
all_model = []
all_pers = []
all_nflics = []

# loop
for file_path in tqdm(filtered_files, desc="Computing AUC"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        continue

    # raw arrays
    gt = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    model = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))
    pers = np.nan_to_num(data["gt0"].cpu().numpy().astype(np.float32))
    nflics = np.nan_to_num(data["nflics"].astype(np.float32))

    # apply neighbourhood max filter to GT and persistence only
    gt = maximum_filter(gt, size=L_pixels)
    pers = maximum_filter(pers, size=L_pixels)

    # threshold GT to binary
    gt = (gt > 0).astype(np.float32)

    # shape check
    for name, arr in zip(
        ["gt", "model", "persistence", "nflics"],
        [gt, model, pers, nflics],
    ):
        if arr.shape != map_shape:
            raise ValueError(f"{name} has shape {arr.shape}, expected {map_shape}")

    # flatten
    all_gt.append(gt.reshape(-1))
    all_model.append(model.reshape(-1))
    all_pers.append(pers.reshape(-1))
    all_nflics.append(nflics.reshape(-1))  # DO NOT filter nflics again!

# concatenate all
all_gt = np.concatenate(all_gt)
all_model = np.concatenate(all_model)
all_pers = np.concatenate(all_pers)
all_nflics = np.concatenate(all_nflics)

# compute AUCs
auc_model = roc_auc_score(all_gt, all_model)
auc_persistence = roc_auc_score(all_gt, all_pers)
auc_nflics = roc_auc_score(all_gt, all_nflics)

# save
np.save(os.path.join(output_dir, f"auc_model_hour_{target_hour}_t{lead_time}.npy"), auc_model)
np.save(os.path.join(output_dir, f"auc_persistence_hour_{target_hour}_t{lead_time}.npy"), auc_persistence)
np.save(os.path.join(output_dir, f"auc_nflics_hour_{target_hour}_t{lead_time}.npy"), auc_nflics)

print(f"\nSaved AUC for t+{lead_time}, hour={target_hour} UTC to {output_dir}")
print(f"Model AUC:       {auc_model:.4f}")
print(f"Persistence AUC: {auc_persistence:.4f}")
print(f"NFLICS AUC:      {auc_nflics:.4f}")
