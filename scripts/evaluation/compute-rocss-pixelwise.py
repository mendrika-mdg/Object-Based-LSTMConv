import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import scipy.ndimage as nd
from sklearn.metrics import roc_auc_score

lead_time = sys.argv[1]
target_hour = sys.argv[2]

base_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/predictions/ncast-nflics-full/t{lead_time}"
output_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/ncast-nflics-full/rocss"
os.makedirs(output_dir, exist_ok=True)

H, W = 512, 512
L_pixels = 9

# storage for all samples
pred_stack = []
ref_stack  = []
gt_stack   = []

def extract_hour(path):
    name = os.path.basename(path)
    parts = name.split("_")
    if len(parts) < 3:
        return None
    hh = parts[2].replace(".pt", "")[:2]
    return hh if hh.isdigit() else None

all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

filtered = [p for p in all_files if extract_hour(p) == target_hour]
print(f"Found {len(filtered)} files at hour={target_hour} UTC")

for file_path in tqdm(filtered, desc="Collecting pixel samples"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception:
        continue

    gt   = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    pred = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))
    nfl  = np.nan_to_num(data["nflics"].astype(np.float32))

    if gt.shape != (H, W):
        continue

    gt   = np.clip(gt, 0, 1)
    pred = np.clip(pred, 0, 1)
    nfl  = np.clip(nfl, 0, 1)

    gt_smooth = nd.maximum_filter(gt, size=L_pixels).astype(np.float32)

    pred_stack.append(pred)
    ref_stack.append(nfl)
    gt_stack.append(gt_smooth)

pred_stack = np.stack(pred_stack, axis=0)
ref_stack  = np.stack(ref_stack,  axis=0)
gt_stack   = np.stack(gt_stack,  axis=0)

rocss_map = np.full((H, W), np.nan, dtype=np.float32)

for i in range(H):
    for j in range(W):
        y = gt_stack[:, i, j]
        p_model = pred_stack[:, i, j]
        p_ref   = ref_stack[:, i, j]

        if np.all(y == 0) or np.all(y == 1):
            continue

        try:
            auc_m = roc_auc_score(y, p_model)
            auc_r = roc_auc_score(y, p_ref)
        except Exception:
            continue

        if auc_m < 1e-6:
            continue

        rocss_map[i, j] = 1 - (auc_r / auc_m)

out_file = os.path.join(
    output_dir,
    f"rocss_pixelwise_hour_{target_hour}_t{lead_time}.npy"
)

np.save(out_file, rocss_map)

print(f"Saved pixelwise ROCSS map to {out_file}")
