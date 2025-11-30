import sys
import numpy as np
import scipy.ndimage as nd
import torch 
import warnings
warnings.filterwarnings("ignore")
import os
from scipy.ndimage import gaussian_filter

from sklearn.isotonic import IsotonicRegression
from joblib import dump   # added

lead_time = sys.argv[1]

val_path = f"/work/scratch-nopw2/mendrika/OB/evaluation/calibration/ncast-nflics/val/t{lead_time}"

out_dir = "/work/scratch-nopw2/mendrika/OB/evaluation/calibration/ncast-nflics"

os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"isotonic_t{lead_time}.joblib")

def count_cores(mask):
    labelled, n = nd.label(mask > 0.5)
    return n

max_samples = 500_000       # recommended
min_prob = 0.01             # optional but improves convection calibration

all_p = []
all_y = []

for root, _, files in os.walk(val_path):
    for fname in files:
        if not fname.endswith(".pt"):
            continue

        fpath = os.path.join(root, fname)
        data = torch.load(fpath, map_location="cpu", weights_only=False)

        mean = data["mean"].detach().cpu().numpy().astype(np.float32)
        gt = data["gt"].detach().cpu().numpy().astype(np.uint8)

        ncores = count_cores(gt)
        if ncores < 15:
            continue

        p_flat = mean.ravel()
        y_flat = gt.ravel()

        if min_prob > 0.0:
            mask = p_flat >= min_prob
            p_flat = p_flat[mask]
            y_flat = y_flat[mask]

        all_p.append(p_flat)
        all_y.append(y_flat)

p = np.concatenate(all_p)
y = np.concatenate(all_y)

n = p.shape[0]
print(f"Total pixels before subsampling: {n}")

if n > max_samples:
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=max_samples, replace=False)
    p = p[idx]
    y = y[idx]
    print(f"Subsampled to {max_samples} pixels for fitting")

print("Fitting isotonic regression...")
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p, y)

print(f"Saving calibrator to {out_path}")
dump(iso, out_path)

print("Done")
