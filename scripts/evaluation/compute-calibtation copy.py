import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

lead_time = sys.argv[1]
target_hour = sys.argv[2]

base_dir = f"/work/scratch-nopw2/mendrika/OB/predictions/t{lead_time}"

n_bins = 11
bins = np.linspace(0, 1, n_bins)

all_p_model = []
all_p_pers = []
all_obs = []

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
    return parts[2].replace(".pt", "")[:2]

filtered_files = [p for p in all_files if extract_hour(p) == target_hour]

for path in tqdm(filtered_files, desc="Loading predictions"):
    try:
        data = torch.load(path, weights_only=False)
    except:
        continue

    gt = data["gt"].cpu().numpy().astype(np.float32)
    model = data["mean"].cpu().numpy().astype(np.float32)
    pers = data["gt0"].cpu().numpy().astype(np.float32)

    gt = np.clip(gt, 0, 1)
    model = np.clip(model, 0, 1)
    pers = np.clip(pers, 0, 1)

    all_p_model.append(model.reshape(-1))
    all_p_pers.append(pers.reshape(-1))
    all_obs.append(gt.reshape(-1))

all_p_model = np.concatenate(all_p_model)
all_p_pers = np.concatenate(all_p_pers)
all_obs = np.concatenate(all_obs)

bin_centres = 0.5 * (bins[1:] + bins[:-1])

def reliability_curve(preds, obs):
    counts = np.zeros(n_bins - 1)
    obs_freq = np.zeros(n_bins - 1)
    for i in range(n_bins - 1):
        mask = (preds >= bins[i]) & (preds < bins[i+1])
        if np.sum(mask) > 0:
            counts[i] = np.sum(mask)
            obs_freq[i] = np.mean(obs[mask])
        else:
            counts[i] = 0
            obs_freq[i] = np.nan
    return obs_freq

rel_model = reliability_curve(all_p_model, all_obs)
rel_pers = reliability_curve(all_p_pers, all_obs)

plt.figure(figsize=(7,7))
plt.plot([0,1], [0,1], linestyle="--", color="grey", label="Perfect reliability")
plt.plot(bin_centres, rel_model, marker="o", label="Model")
plt.plot(bin_centres, rel_pers, marker="x", label="Persistence")
plt.xlabel("Forecast probability")
plt.ylabel("Observed frequency")
plt.title(f"Reliability Diagram (t+{lead_time}, hour={target_hour})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
