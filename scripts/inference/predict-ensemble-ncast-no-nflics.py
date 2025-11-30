import os
import re
import sys
import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
from netCDF4 import Dataset

import warnings
warnings.filterwarnings("ignore")

# model path
sys.path.append("/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/training")
from ncast import Core2MapModel

# arguments
LEAD_TIME = sys.argv[1]
YEAR  = sys.argv[2]
MONTH = sys.argv[3]
HOUR  = sys.argv[4]

# paths
ENSEMBLE_DIR = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/OB/checkpoints/WS/transformer/t{LEAD_TIME}"
SCALER_PATH  = "/home/users/mendrika/Object-Based-LSTMConv/outputs/scaler/scaler_realcores.pt"
INPUT_ROOT   = "/work/scratch-nopw2/mendrika/OB/raw/inputs_t0"
OUTPUT_BASE = f"/work/scratch-nopw2/mendrika/OB/evaluation/calibration/ncast-nflics/val/t{LEAD_TIME}"

os.makedirs(OUTPUT_BASE, exist_ok=True)
OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"{YEAR}{MONTH}", f"{HOUR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# device
DEVICE = torch.device("cpu")

# threads
num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
torch.set_num_threads(num_threads)
print(f"Running on CPU with {num_threads} threads")

# load ensemble models
ckpts = []
for root, _, files in os.walk(ENSEMBLE_DIR):
    for f in files:
        if f.endswith(".ckpt"):
            ckpts.append(os.path.join(root, f))
ckpts = sorted(ckpts)
if not ckpts:
    raise RuntimeError(f"No checkpoints in {ENSEMBLE_DIR}")

models = []
for path in ckpts:
    model = Core2MapModel.load_from_checkpoint(path, map_location=DEVICE)
    model.eval()
    models.append(model)
print(f"Loaded {len(models)} ensemble models on CPU")

# ensemble inference without MC dropout
def ensemble_predict(models, x):
    preds = []
    with torch.no_grad():
        for model in models:
            pred = torch.sigmoid(model(x)).squeeze(0).squeeze(0)
            preds.append(pred)
    preds = torch.stack(preds)
    mean_pred = preds.mean(dim=0)
    var_pred  = preds.var(dim=0)
    return mean_pred, var_pred

# scaler
MASK_COL_INDEX = 12
COLS_TO_SCALE = range(4, 12)
scaler = torch.load(SCALER_PATH, map_location="cpu", weights_only=False)
mean = np.asarray(scaler["mean"])
scale = np.asarray(scaler["scale"])

# regex
pattern = re.compile(r"input-(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\.pt$")

def load_zcast_input(y, mo, d, h, mi):
    path = f"{INPUT_ROOT}/input-{y}{mo}{d}_{h}{mi}.pt"
    return torch.load(path)

def load_output(y, mo, d, h, mi, lead):
    gt_path   = f"/work/scratch-nopw2/mendrika/OB/raw/targets_t{lead}/target-{y}{mo}{d}_{h}{mi}.pt"
    pers_path = f"/work/scratch-nopw2/mendrika/OB/raw/targets_t0/target-{y}{mo}{d}_{h}{mi}.pt"
    gt = torch.load(gt_path)["data"].numpy()
    persistence = torch.load(pers_path)["data"].numpy()
    return gt, persistence

# discover input files
input_files = []
for f in sorted(os.listdir(INPUT_ROOT)):
    m = pattern.match(f)
    if m:
        y, mo, d, h, mi = m.groups()
        if y == YEAR and mo == MONTH and h == HOUR:
            input_files.append((y, mo, d, h, mi))

print(f"Detected {len(input_files)} inputs for {YEAR}-{MONTH} {HOUR} UTC")

# inference loop
for year, month, day, hour, minute in tqdm(input_files, desc="Predicting"):
    try:
        data = load_zcast_input(year, month, day, hour, minute)
        gt, persistence = load_output(year, month, day, hour, minute, LEAD_TIME)
        input_tensor = data["input_tensor"].clone().unsqueeze(0)

        X = input_tensor[0]
        X_np = X.numpy()
        flat = X_np.reshape(-1, X_np.shape[-1])
        flat[:, COLS_TO_SCALE] = (flat[:, COLS_TO_SCALE] - mean) / scale

        X_scaled = torch.tensor(flat.reshape(X_np.shape), dtype=torch.float32)
        input_scaled = X_scaled.unsqueeze(0).to(DEVICE)

        mean_pred, var_pred = ensemble_predict(models, input_scaled)

        out_file = os.path.join(
            OUTPUT_DIR,
            f"pred_{year}{month}{day}_{hour}{minute}.pt"
        )

        torch.save({
            "mean": mean_pred.cpu(),
            "var": var_pred.cpu(),
            "gt": torch.tensor(gt),
            "gt0": torch.tensor(persistence),
        }, out_file)

    except Exception as e:
        print(f"Skipping {year}-{month}-{day} {hour}:{minute}: {e}")

print("All ensemble nowcasts completed on CPU.")
