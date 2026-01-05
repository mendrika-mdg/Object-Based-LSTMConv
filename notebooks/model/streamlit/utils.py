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

import sys
sys.path.append("/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/training")
from pancast import Core2MapModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Africa crop
y_min, y_max = 48, 2062
x_min, x_max = 81, 2267

# extent = (-18, 51, -35, 24)

# Map bounds
BOUNDS = [
    [-35.0, -18.0],   # lat_min, lon_min
    [ 24.0,  51.0],   # lat_max, lon_max
]

ENSEMBLE_DIRS = {
    30:  "/gws/nopw/j04/wiser_ewsa/mrakotomanga/pancast/final/checkpoints/t030min",
    60:  "/gws/nopw/j04/wiser_ewsa/mrakotomanga/pancast/final/checkpoints/t060min",
    90:  "/gws/nopw/j04/wiser_ewsa/mrakotomanga/pancast/final/checkpoints/t090min",
    120: "/gws/nopw/j04/wiser_ewsa/mrakotomanga/pancast/final/checkpoints/t120min",
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

    base = f"/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/{t['year']}/{t['month']}"
    fname = f"{t['year']}{t['month']}{t['day']}{t['hour']}{t['minute']}.nc"

    ds = Dataset(os.path.join(base, fname))
    cores = ds["cores"][0, y_min:y_max+1, x_min:x_max+1]
    ds.close()

    return cores != 0


def load_input_tensor(time_dict):
    fname = f"input-{time_dict['year']}{time_dict['month']}{time_dict['day']}_{time_dict['hour']}{time_dict['minute']}.pt"

    paths = [
        "/work/scratch-nopw2/mendrika/pancast/raw/inputs_t0",
        "/gws/nopw/j04/wiser_ewsa/mrakotomanga/pancast/raw/inputs_t0",
    ]

    for base in paths:
        f = os.path.join(base, fname)
        if os.path.exists(f):
            return torch.load(f)["input_tensor"]

    raise FileNotFoundError(fname)


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
