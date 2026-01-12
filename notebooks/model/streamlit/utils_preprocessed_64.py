import os
import torch
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image
import base64
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

import sys
sys.path.append("/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/training")
from pancast_64 import Core2MapModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


BOUNDS = [
    [-35.0, -18.0],
    [24.0, 51.0],
]

ENSEMBLE_DIRS = {
    30:  "/gws/nopw/j04/wiser_ewsa/mrakotomanga/pancast_64/checkpoints/t030min",
    60:  "/gws/nopw/j04/wiser_ewsa/mrakotomanga/pancast_64/checkpoints/t060min",
    90:  "/gws/nopw/j04/wiser_ewsa/mrakotomanga/pancast_64/checkpoints/t090min",
    120: "/gws/nopw/j04/wiser_ewsa/mrakotomanga/pancast_64/checkpoints/t120min",
}

SCALER_PATH = "/home/users/mendrika/Object-Based-LSTMConv/outputs/scaler-africa/scaler_realcores_online.pt"
COLS_TO_SCALE = range(4, 12)


def add_minutes(time_dict, minutes):
    t = datetime(
        int(time_dict["year"]),
        int(time_dict["month"]),
        int(time_dict["day"]),
        int(time_dict["hour"]),
        int(time_dict["minute"]),
    ) + timedelta(minutes=minutes)

    return {
        "year": f"{t.year:04d}",
        "month": f"{t.month:02d}",
        "day": f"{t.day:02d}",
        "hour": f"{t.hour:02d}",
        "minute": f"{t.minute:02d}",
    }


def load_ground_truth(time_dict, lead_time):
    
    t = add_minutes(time_dict, lead_time)

    if int(t["year"]) <= 2024:

        base = f"/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/{t['year']}/{t['month']}"
        fname = f"{t['year']}{t['month']}{t['day']}{t['hour']}{t['minute']}.nc"

        ds = Dataset(os.path.join(base, fname))

        y_min, y_max = 48, 2062
        x_min, x_max = 81, 2267

        cores = ds["cores"][0, y_min:y_max+1, x_min:x_max+1]
        ds.close()

        return cores != 0
    else:
        y_min, y_max = 48, 2062
        x_min, x_max = 77, 2262

        base = f"/gws/ssde/j25b/swift/rt_cores/{t['year']}/{t['month']}/{t['day']}/{t['hour']}{t['minute']}"
        fname = f"Convective_struct_extended_{t['year']}{t['month']}{t['day']}{t['hour']}{t['minute']}_000.nc"

        ds = Dataset(os.path.join(base, fname))
        cores = ds["cores"][y_min:y_max+1, x_min:x_max+1]
        ds.close()

        return cores < 0


def load_input_tensor(time_dict):
    fname = (
        f"input-{time_dict['year']}"
        f"{time_dict['month']}"
        f"{time_dict['day']}_"
        f"{time_dict['hour']}"
        f"{time_dict['minute']}.pt"
    )

    if int(time_dict["year"]) <= 2024:
        base = "/gws/ssde/j25b/swift/mendrika/pancast/raw/inputs_t0"
    else:
        base = "/gws/ssde/j25b/swift/mendrika/pancast/raw/2025/inputs_t0"

    path = os.path.join(base, fname)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return torch.load(path)["input_tensor"]


def load_models(lead_time):
    models = []
    for seed in sorted(os.listdir(ENSEMBLE_DIRS[lead_time])):
        ckpt = os.path.join(
            ENSEMBLE_DIRS[lead_time],
            seed,
            "lr7e-05/best-pancast.ckpt",
        )
        if os.path.exists(ckpt):
            m = Core2MapModel.load_from_checkpoint(ckpt, map_location=DEVICE)
            m.eval().to(DEVICE)
            models.append(m)
    return models


def scale_input(X):
    scaler = torch.load(SCALER_PATH, weights_only=False)
    mean = np.asarray(scaler["mean"])
    scale = np.asarray(scaler["scale"])

    X_np = X.numpy()
    flat = X_np.reshape(-1, X_np.shape[-1])
    flat[:, COLS_TO_SCALE] = (flat[:, COLS_TO_SCALE] - mean) / scale

    return torch.tensor(flat.reshape(X_np.shape), dtype=torch.float32)


def ensemble_predict(models, x):
    preds = []
    with torch.no_grad():
        for m in models:
            p = torch.sigmoid(m(x)).squeeze()
            preds.append(p)
    return torch.stack(preds).mean(dim=0)


def rescale_after_threshold(pred, floor=0.05, eps=1e-6):
    pred = pred.copy()
    pred[pred < floor] = 0.0

    max_val = pred.max()
    if max_val > eps:
        pred = pred / max_val

    return pred


def smooth_prediction(pred, sigma=1.0):
    return gaussian_filter(pred, sigma=sigma)


def bin_probabilities(pred, bins=(0.05, 0.10, 0.20, 0.30)):
    out = np.zeros_like(pred)

    out[(pred >= bins[0]) & (pred < bins[1])] = 0.25
    out[(pred >= bins[1]) & (pred < bins[2])] = 0.50
    out[(pred >= bins[2]) & (pred < bins[3])] = 0.75
    out[pred >= bins[3]] = 1.00

    return out

def array_to_rgba_overlay(data, mask, cmap_name, vmin, vmax, alpha):
    data = np.nan_to_num(data)
    data = np.flipud(data)
    mask = np.flipud(mask.astype(bool))

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm(data))
    rgba[..., -1] = np.where(mask, alpha, 0.0)

    img = (rgba * 255).astype(np.uint8)
    pil = Image.fromarray(img, mode="RGBA")

    buf = BytesIO()
    pil.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def gamma_boost(pred, gamma=2.0):
    pred = pred.copy()
    pred = np.clip(pred, 0.0, 1.0)
    return pred ** gamma
