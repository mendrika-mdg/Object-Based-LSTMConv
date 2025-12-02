import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# arguments
lead_time = sys.argv[1]

# directories
base_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/predictions/ncast-calibrated/t{lead_time}"
output_dir = f"/home/users/mendrika/Object-Based-LSTMConv/outputs/evaluation/ncast-calibrated/roc"
os.makedirs(output_dir, exist_ok=True)

# thresholds for ROC
thresholds = np.linspace(0, 1, 101)

# counters for RAW model
TP_model = np.zeros_like(thresholds, dtype=np.int64)
FP_model = np.zeros_like(thresholds, dtype=np.int64)
TN_model = np.zeros_like(thresholds, dtype=np.int64)
FN_model = np.zeros_like(thresholds, dtype=np.int64)

# counters for persistence
TP_pers = np.zeros_like(thresholds, dtype=np.int64)
FP_pers = np.zeros_like(thresholds, dtype=np.int64)
TN_pers = np.zeros_like(thresholds, dtype=np.int64)
FN_pers = np.zeros_like(thresholds, dtype=np.int64)

# collect all files
all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt"):
            all_files.append(os.path.join(root, f))

print(f"Found {len(all_files)} files")

# streaming loop
for file_path in tqdm(all_files, desc=f"Streaming ROC for t+{lead_time}"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception:
        continue

    gt = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    pred_raw = np.nan_to_num(data["mean"].cpu().numpy().astype(np.float32))    # RAW
    pers = np.nan_to_num(data["gt0"].cpu().numpy().astype(np.float32))

    # flatten
    gt = (gt.reshape(-1) > 0).astype(np.int8)
    pred_raw = pred_raw.reshape(-1)
    pers = pers.reshape(-1)

    # thresholds loop
    for i, th in enumerate(thresholds):

        # raw model
        pred_bin = pred_raw >= th
        TP_model[i] += np.sum((pred_bin == 1) & (gt == 1))
        FP_model[i] += np.sum((pred_bin == 1) & (gt == 0))
        TN_model[i] += np.sum((pred_bin == 0) & (gt == 0))
        FN_model[i] += np.sum((pred_bin == 0) & (gt == 1))

        # persistence
        pers_bin = pers >= th
        TP_pers[i] += np.sum((pers_bin == 1) & (gt == 1))
        FP_pers[i] += np.sum((pers_bin == 1) & (gt == 0))
        TN_pers[i] += np.sum((pers_bin == 0) & (gt == 0))
        FN_pers[i] += np.sum((pers_bin == 0) & (gt == 1))

# compute TPR and FPR
TPR_model = TP_model / (TP_model + FN_model + 1e-12)
FPR_model = FP_model / (FP_model + TN_model + 1e-12)

TPR_pers = TP_pers / (TP_pers + FN_pers + 1e-12)
FPR_pers = FP_pers / (FP_pers + TN_pers + 1e-12)

# save results
np.save(os.path.join(output_dir, f"roc_thresholds_t{lead_time}.npy"), thresholds)
np.save(os.path.join(output_dir, f"roc_fpr_model_t{lead_time}.npy"), FPR_model)
np.save(os.path.join(output_dir, f"roc_tpr_model_t{lead_time}.npy"), TPR_model)
np.save(os.path.join(output_dir, f"roc_fpr_persistence_t{lead_time}.npy"), FPR_pers)
np.save(os.path.join(output_dir, f"roc_tpr_persistence_t{lead_time}.npy"), TPR_pers)

print("\nROC arrays saved in:", output_dir)
print("Done.")
