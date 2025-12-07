import os
import sys
import torch
import numpy as np
import pandas as pd
import scipy.ndimage as nd
from tqdm import tqdm

lead_time = 1

base_dir = f"/work/scratch-nopw2/mendrika/OB/evaluation/predictions/ncast-calibrated/t{lead_time}"
output_csv = f"/home/users/mendrika/Object-Based-LSTMConv/outputs/evaluation/ncast-calibrated/core_counts_t{lead_time}.csv"

def count_cores(mask):
    labelled, n = nd.label(mask > 0.5)
    return int(n)

all_files = []
for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".pt") and f.startswith("pred_"):
            all_files.append(os.path.join(root, f))

rows = []

for file_path in tqdm(all_files, desc=f"Counting cores t+{lead_time}"):
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception:
        continue

    try:
        gt = np.nan_to_num(data["gt"].cpu().numpy().astype(np.float32))
    except Exception:
        continue

    ncores = count_cores(gt)
    if ncores < 15:
        continue

    name = os.path.basename(file_path).replace(".pt", "")
    parts = name.split("_")
    if len(parts) != 3:
        continue

    date = parts[1]
    time = parts[2]

    year = int(date[0:4])
    month = int(date[4:6])
    day = int(date[6:8])
    hour = int(time[0:2])
    minute = int(time[2:4])

    rows.append({
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "ncores": ncores,
    })

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)

print(f"\nSaved {len(df)} rows to {output_csv}")
