import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import scipy.ndimage as nd

# arguments
lead_time = sys.argv[1]

# directories
base_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/predictions/ncast-nflics-full-corrected/t{lead_time}"
output_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/ncast-nflics-full-corrected/roc"
os.makedirs(output_dir, exist_ok=True)

H, W = 512, 512
L_pixels = 8
thresholds = np.linspace(0, 1, 101)

# accumulators ---- model
TP_model = np.zeros_like(thresholds, dtype=np.int64)
FP_model = np.zeros_like(thresholds, dtype=np.int64)
TN_model = np.zeros_like(thresholds, dtype=np.int64)
FN_model = np.zeros_like(thresholds, dtype=np.int64)

# accumulators ---- persistence
TP_pers = np.zeros_like(thresholds, dtype=np.int64)
FP_pers = np.zeros_like(thresholds, dtype=np.int64)
TN_pers = np.zeros_like(thresholds, dtype=np.int64)
FN_pers = np.zeros_like(thresholds, dtype=np.int64)

# accumulators ---- NFLICS
TP_nfl = np.zeros_like(thresholds, dtype=np.int64)
FP_nfl = np.zeros_like(thresholds, dtype=np.int64)
TN_nfl = np.zeros_like(thresholds, dtype=np.int64)
FN_nfl = np.zeros_like(thresholds, dtype=np.int64)

# collect files
all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

print(f"Found {len(all_files)} files")

used_scenes = 0

# streaming loop
for file_path in tqdm(all_files, desc=f"Streaming ROC for t+{lead_time}"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception:
        continue

    gt = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    model = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))
    pers  = np.nan_to_num(data["gt0"].cpu().numpy().astype(np.float32))
    nfl   = np.nan_to_num(data["nflics"].astype(np.float32))

    if gt.shape != (H, W):
        continue

    used_scenes += 1

    # smooth ground truth
    gt_smooth = nd.maximum_filter(gt, size=L_pixels).astype(np.float32)
    gt_bin = (gt_smooth.reshape(-1) > 0).astype(np.int8)

    # flatten predictions
    model = np.clip(model, 0, 1).reshape(-1)
    pers  = np.clip(pers,  0, 1).reshape(-1)
    nfl   = np.clip(nfl,   0, 1).reshape(-1)

    # thresholding
    for i, th in enumerate(thresholds):

        # model
        m = model >= th
        TP_model[i] += np.sum((m == 1) & (gt_bin == 1))
        FP_model[i] += np.sum((m == 1) & (gt_bin == 0))
        TN_model[i] += np.sum((m == 0) & (gt_bin == 0))
        FN_model[i] += np.sum((m == 0) & (gt_bin == 1))

        # persistence
        p = pers >= th
        TP_pers[i] += np.sum((p == 1) & (gt_bin == 1))
        FP_pers[i] += np.sum((p == 1) & (gt_bin == 0))
        TN_pers[i] += np.sum((p == 0) & (gt_bin == 0))
        FN_pers[i] += np.sum((p == 0) & (gt_bin == 1))

        # nfl
        n = nfl >= th
        TP_nfl[i] += np.sum((n == 1) & (gt_bin == 1))
        FP_nfl[i] += np.sum((n == 1) & (gt_bin == 0))
        TN_nfl[i] += np.sum((n == 0) & (gt_bin == 0))
        FN_nfl[i] += np.sum((n == 0) & (gt_bin == 1))

print(f"Used {used_scenes} scenes")

# compute ROC curves
TPR_model = TP_model / (TP_model + FN_model + 1e-12)
FPR_model = FP_model / (FP_model + TN_model + 1e-12)

TPR_pers = TP_pers / (TP_pers + FN_pers + 1e-12)
FPR_pers = FP_pers / (FP_pers + TN_pers + 1e-12)

TPR_nfl = TP_nfl / (TP_nfl + FN_nfl + 1e-12)
FPR_nfl = FP_nfl / (FP_nfl + TN_nfl + 1e-12)

# save arrays
np.save(os.path.join(output_dir, f"roc_thresholds_t{lead_time}.npy"), thresholds)

np.save(os.path.join(output_dir, f"roc_fpr_model_t{lead_time}.npy"), FPR_model)
np.save(os.path.join(output_dir, f"roc_tpr_model_t{lead_time}.npy"), TPR_model)

np.save(os.path.join(output_dir, f"roc_fpr_persistence_t{lead_time}.npy"), FPR_pers)
np.save(os.path.join(output_dir, f"roc_tpr_persistence_t{lead_time}.npy"), TPR_pers)

np.save(os.path.join(output_dir, f"roc_fpr_nflics_t{lead_time}.npy"), FPR_nfl)
np.save(os.path.join(output_dir, f"roc_tpr_nflics_t{lead_time}.npy"), TPR_nfl)

print("\nROC arrays saved in:", output_dir)
print("Done.")
