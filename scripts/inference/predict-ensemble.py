import os
import re
import sys
import torch
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# model path
sys.path.append("/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/training")
from obconvlstm_highres_small import OB2MapModel

# arguments
LEAD_TIME = sys.argv[1]
YEAR  = sys.argv[2]
MONTH = sys.argv[3]
HOUR  = sys.argv[4]

# paths
ENSEMBLE_DIR = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/OB/checkpoints/lstm_core2map_hybrid/latent32_fss3/t{LEAD_TIME}"
SCALER_PATH  = "/home/users/mendrika/Object-Based-LSTMConv/outputs/scaler/scaler_realcores.pt"
INPUT_ROOT   = "/work/scratch-nopw2/mendrika/OB/raw/inputs_t0"
OUTPUT_BASE  = f"/work/scratch-nopw2/mendrika/OB/predictions/t{LEAD_TIME}"

# output directory
os.makedirs(OUTPUT_BASE, exist_ok=True)
OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"{YEAR}{MONTH}", f"{HOUR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# device
DEVICE = torch.device("cpu")

# threads
num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
torch.set_num_threads(num_threads)
print(f"Running on CPU with {num_threads} threads")

# mc samples
MC_SAMPLES = int(os.environ.get("MC_SAMPLES", 10))

# load ensemble
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
    model = OB2MapModel.load_from_checkpoint(path, map_location=DEVICE)
    model.eval()
    models.append(model)
print(f"Loaded {len(models)} ensemble models on CPU")

# enable dropout
def enable_dropout(model, p=0.05):
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()
            m.p = p

# ensemble inference
def ensemble_mc_predict(models, x):
    preds_all = []
    for model in models:
        enable_dropout(model, p=0.05)
        mc_preds = []
        with torch.no_grad():
            for _ in range(MC_SAMPLES):
                pred = torch.sigmoid(model(x)).squeeze(0).squeeze(0)
                mc_preds.append(pred)
        preds_all.append(torch.stack(mc_preds))
    preds_all = torch.stack(preds_all)
    mean_pred = preds_all.mean(dim=(0, 1))
    var_pred  = preds_all.var(dim=(0, 1))
    return mean_pred, var_pred

# scaler
MASK_COL_INDEX = 12
COLS_TO_SCALE = range(4, 12)
scaler = torch.load(SCALER_PATH, map_location="cpu", weights_only=False)
mean = np.asarray(scaler["mean"])
scale = np.asarray(scaler["scale"])

# regex
pattern = re.compile(r"input-(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\.pt$")

# load input
def load_zcast_input(y, mo, d, h, mi, lead):
    path = f"{INPUT_ROOT}/input-{y}{mo}{d}_{h}{mi}.pt"
    return torch.load(path)

# load outputs (gt_t and persistence)
def load_output(y, mo, d, h, mi, lead):
    gt_path = f"/work/scratch-nopw2/mendrika/OB/raw/targets_t{lead}/target-{y}{mo}{d}_{h}{mi}.pt"
    pers_path = f"/work/scratch-nopw2/mendrika/OB/raw/targets_t0/target-{y}{mo}{d}_{h}{mi}.pt"
    gt = torch.load(gt_path)["data"].numpy()
    persistence = torch.load(pers_path)["data"].numpy()
    return gt, persistence

# find matching files
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
        # load input
        data = load_zcast_input(year, month, day, hour, minute, LEAD_TIME)

        # load outputs
        gt, persistence = load_output(year, month, day, hour, minute, LEAD_TIME)

        # extract tensor, add batch dim
        input_tensor = data["input_tensor"].clone().unsqueeze(0)

        # remove batch
        X = input_tensor[0]

        # to numpy
        X_np = X.numpy()

        # flatten cores
        flat = X_np.reshape(-1, X_np.shape[-1])

        # scale
        flat[:, COLS_TO_SCALE] = (flat[:, COLS_TO_SCALE] - mean) / scale

        # restore shape
        X_scaled = torch.tensor(flat.reshape(X_np.shape), dtype=torch.float32)

        # add batch
        input_scaled = X_scaled.unsqueeze(0).to(DEVICE)

        # inference
        mean_pred, var_pred = ensemble_mc_predict(models, input_scaled)

        # output file path
        out_file = os.path.join(
            OUTPUT_DIR,
            f"pred_{year}{month}{day}_{hour}{minute}.pt"
        )

        # save
        torch.save({
            "mean": mean_pred.cpu(),
            "var": var_pred.cpu(),
            "gt": torch.tensor(gt),
            "gt0": torch.tensor(persistence)
        }, out_file)

    except Exception as e:
        print(f"Skipping {year}-{month}-{day} {hour}:{minute}: {e}")

print("All ensemble nowcasts completed on CPU.")
