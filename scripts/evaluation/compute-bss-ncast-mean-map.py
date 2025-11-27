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

# accumulate pixelwise BS
bs_model_sum = np.zeros(map_shape, dtype=np.float64)
bs_ref_sum   = np.zeros(map_shape, dtype=np.float64)
count        = np.zeros(map_shape, dtype=np.int32)

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

# loop through files
for file_path in tqdm(filtered_files, desc="Computing pixelwise BS"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        continue

    gt       = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    model    = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))
    persist  = np.nan_to_num(data["gt0"].cpu().numpy().astype(np.float32))

    if gt.shape != map_shape:
        raise ValueError(f"Bad shape {gt.shape}, expected {map_shape}")

    # compute BS per pixel
    bs_model = (np.clip(model,   0,1) - gt)**2
    bs_ref   = (np.clip(persist, 0,1) - gt)**2

    # RESTRICTED mask (same logic as before)
    trivial_mask = bs_ref > 1e-12
    conv_mask = (gt > 0) | (model > 0.1) | (persist > 0)
    valid = trivial_mask & conv_mask

    # accumulate only valid pixels
    bs_model_sum[valid] += bs_model[valid]
    bs_ref_sum[valid]   += bs_ref[valid]
    count[valid]        += 1

# avoid division by zero
valid_global = count > 0

pixelwise_bss = np.full(map_shape, np.nan, dtype=np.float32)
pixelwise_bss[valid_global] = 1 - (bs_model_sum[valid_global] /
                                   (bs_ref_sum[valid_global] + 1e-12))

# save the pixelwise BSS map
out_path = os.path.join(output_dir, f"pixelwise_bss_hour_{target_hour}_t{lead_time}.npy")
np.save(out_path, pixelwise_bss)

print(f"Saved pixelwise BSS map → {out_path}")
print("Valid pixels:", np.sum(valid_global))
