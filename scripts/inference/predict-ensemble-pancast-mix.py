import os
import re
import sys
import torch
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.append("/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/training")
from pancast_64 import Core2MapModel


LEAD_TIME = int(sys.argv[1])
YEAR  = sys.argv[2]
MONTH = sys.argv[3]
HOUR  = sys.argv[4]


ENSEMBLE_DIR = f"/gws/nopw/j04/wiser_ewsa/mrakotomanga/pancast_64/checkpoints/t{LEAD_TIME:03d}min"
SCALER_PATH  = "/home/users/mendrika/Object-Based-LSTMConv/outputs/scaler-africa/scaler_realcores_online.pt"
INPUT_ROOT   = "/gws/ssde/j25b/swift/mendrika/pancast/test/inputs_t0"
GT_ROOT      = f"/gws/ssde/j25b/swift/mendrika/pancast/test/targets_t{LEAD_TIME:03d}min"
OUTPUT_BASE  = f"/gws/ssde/j25b/swift/mendrika/pancast/nowcasts/test/t{LEAD_TIME}"


os.makedirs(OUTPUT_BASE, exist_ok=True)
OUTPUT_DIR = os.path.join(OUTPUT_BASE, f"{YEAR}{MONTH}", f"{HOUR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


DEVICE = torch.device("cpu")


num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
torch.set_num_threads(num_threads)
print(f"CPU threads: {num_threads}")


NY = 2015
NX = 2186


# Load ensemble models

ckpts = []
for root, _, files in os.walk(ENSEMBLE_DIR):
    for f in files:
        if f.endswith(".ckpt"):
            ckpts.append(os.path.join(root, f))

ckpts = sorted(ckpts)

if not ckpts:
    raise RuntimeError(f"No checkpoints found in {ENSEMBLE_DIR}")

models = []
for path in ckpts:
    model = Core2MapModel.load_from_checkpoint(path, map_location=DEVICE)
    model.eval()
    models.append(model)

print(f"Loaded {len(models)} ensemble models")


# Ensemble inference

@torch.inference_mode()
def ensemble_predict(x):

    preds = []

    for model in models:
        p = torch.sigmoid(model(x)).squeeze(0).squeeze(0)
        preds.append(p)

    preds = torch.stack(preds, dim=0)

    return preds.mean(0), preds.var(0)


# Scaler
COLS_TO_SCALE = range(4, 12)

scaler = torch.load(SCALER_PATH, map_location="cpu", weights_only=False)
mean  = torch.tensor(scaler["mean"], dtype=torch.float32)
scale = torch.tensor(scaler["scale"], dtype=torch.float32)


# Helpers

pattern = re.compile(r"input-(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\.pt$")


def load_input(y, m, d, h, mi):
    path = f"{INPUT_ROOT}/input-{y}{m}{d}_{h}{mi}.pt"
    return torch.load(path, map_location="cpu")


def load_gt(y, m, d, h, mi):
    path = f"{GT_ROOT}/target-{y}{m}{d}_{h}{mi}.pt"
    gt = torch.load(path, map_location="cpu")["data"]
    return gt


def force_2015x2186(arr, name):

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(arr.shape)}")

    if arr.shape == (NY, NX):
        return arr

    if arr.shape == (NY, NX + 1):
        return arr[:, :-1]

    raise ValueError(f"{name} shape {tuple(arr.shape)} is not compatible with {(NY, NX)}")


# Discover inputs

input_files = []

for f in sorted(os.listdir(INPUT_ROOT)):
    m = pattern.match(f)
    if m:
        y, mo, d, h, mi = m.groups()
        if y == YEAR and mo == MONTH and h == HOUR:
            input_files.append((y, mo, d, h, mi))

print(f"Found {len(input_files)} inputs for {YEAR}-{MONTH} {HOUR} UTC")

if len(input_files) == 0:
    print(f"No data for {YEAR} - month {MONTH} - and {HOUR}h — exiting cleanly.")
    sys.exit(0)

    
# Inference loop

for y, mo, d, h, mi in tqdm(input_files, desc="Predicting"):

    try:
        data = load_input(y, mo, d, h, mi)
        gt = load_gt(y, mo, d, h, mi)

        X = data["input_tensor"].float()

        flat = X.view(-1, X.shape[-1])
        flat[:, COLS_TO_SCALE] = (flat[:, COLS_TO_SCALE] - mean) / scale

        input_scaled = X.unsqueeze(0)

        mean_pred, var_pred = ensemble_predict(input_scaled)

        mean_pred = force_2015x2186(mean_pred, "pred_mean")
        var_pred  = force_2015x2186(var_pred,  "pred_var")
        gt        = force_2015x2186(gt,       "gt")

        out_file = os.path.join(
            OUTPUT_DIR,
            f"pred_{y}{mo}{d}_{h}{mi}.pt"
        )

        torch.save(
            {
                "mean": mean_pred,
                "var":  var_pred,
                "gt":   gt
            },
            out_file
        )

    except Exception as e:
        print(f"Skipping {y}-{mo}-{d} {h}:{mi} -> {e}")


print("Finished all ensemble nowcasts on CPU.")
