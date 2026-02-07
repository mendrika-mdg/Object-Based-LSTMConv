import sys
import numpy as np
import pandas as pd
from pathlib import Path
from netCDF4 import Dataset

import torch
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, roc_curve

LEAD_TIME = int(sys.argv[1])
RADIUS = int(sys.argv[2])

geodata = np.load("/gws/nopw/j04/cocoon/SSA_domain/lat_lon_2268_2080.npz")
y_min, y_max = 48, 2062
x_min, x_max = 81, 2267

lons = geodata["lon"][y_min:y_max + 1, x_min:x_max + 1]
lats = geodata["lat"][y_min:y_max + 1, x_min:x_max + 1]

Ny, Nx = lats.shape
latlon_points = np.column_stack([lats.ravel(), lons.ravel()])
tree = cKDTree(latlon_points)

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

BASE_PATH = Path("/gws/nopw/j04/wiser_ewsa/mrakotomanga/initiations/txt_op")

COLUMNS = [
    "mo", "dy", "hr", "mn",
    "latN_c", "lonE_c",
    "i_c", "j_c", "i", "j",
    "topo_sd", "wf", "minT", "Rmax"
]

def load_month_file(filepath):
    year = int(filepath.name[5:9])

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=1,
        names=COLUMNS
    )

    df["year"] = year

    df["datetime"] = pd.to_datetime(
        dict(
            year=df["year"],
            month=df["mo"],
            day=df["dy"],
            hour=df["hr"],
            minute=df["mn"]
        ),
        utc=True
    )

    return df

def get_initiations_at_time(timestamp):
    timestamp = pd.to_datetime(timestamp, utc=True)
    filepath = BASE_PATH / f"init_{timestamp:%Y%m}_T40_vn3_30km.txt"

    if not filepath.exists():
        raise FileNotFoundError(f"Missing file: {filepath}")

    df = load_month_file(filepath)
    timestamp = timestamp.floor("15min")
    return df[df["datetime"] == timestamp]

def evaluate_ci_event(
    j0, i0,
    lats, lons,
    core_t0,
    core_truth,
    pancast_prob,
    R_km=30,
    R_excl_km=100,
    half_window=60,
):
    Ny, Nx = lats.shape

    j1 = max(j0 - half_window, 0)
    j2 = min(j0 + half_window + 1, Ny)
    i1 = max(i0 - half_window, 0)
    i2 = min(i0 + half_window + 1, Nx)

    lat_win = lats[j1:j2, i1:i2]
    lon_win = lons[j1:j2, i1:i2]

    dist = haversine_km(
        lats[j0, i0],
        lons[j0, i0],
        lat_win,
        lon_win,
    )

    excl_mask = dist <= R_excl_km
    if np.any(core_t0[j1:j2, i1:i2][excl_mask] == 1):
        return np.nan, np.nan

    match_mask = dist <= R_km

    core_present = np.any(core_truth[j1:j2, i1:i2][match_mask] == 1)
    Pmax = np.nanmax(pancast_prob[j1:j2, i1:i2][match_mask])

    return core_present, Pmax

def load_nowcast_and_gt(year, month, day, hour, minute, lead_time):
    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    lead_time = int(lead_time)

    nowcast_root = f"/gws/ssde/j25b/swift/mendrika/pancast/nowcasts/t{lead_time:03d}min"

    fpath = (
        f"{nowcast_root}/"
        f"{year}{month:02d}/"
        f"{hour:02d}/"
        f"pred_{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}.pt"
    )

    data = torch.load(fpath, map_location="cpu")
    nowcast = data["mean"].numpy()
    gt = data["gt"].numpy()

    return nowcast, gt

parent = Path(f"/gws/ssde/j25b/swift/mendrika/pancast/nowcasts/t{LEAD_TIME:03d}min")
pt_paths = list(parent.rglob("pred_*.pt"))

print("Number of pt files:", len(pt_paths), flush=True)

all_labels = []
all_scores = []

for f in pt_paths:
    date = f.stem.split("_")[1]
    time = f.stem.split("_")[2]

    year, month, day = date[0:4], date[4:6], date[6:8]
    hour, minute = time[0:2], time[2:4]

    print(year, month, day, hour, minute, flush=True)

    valid_time = pd.Timestamp(
        year=int(year),
        month=int(month),
        day=int(day),
        hour=int(hour),
        minute=int(minute),
        tz="UTC",
    )

    try:
        nowcast, gt = load_nowcast_and_gt(year, month, day, hour, minute, LEAD_TIME)
    except FileNotFoundError:
        print("Missing nowcast:", f, flush=True)
        continue

    path_core_t0 = f"/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/{year}/{month}"
    file_t0 = f"{path_core_t0}/{year}{month}{day}{hour}{minute}.nc"

    if not Path(file_t0).exists():
        print("Missing core file:", file_t0, flush=True)
        continue

    with Dataset(file_t0, mode="r") as ds:
        cores_t0 = ds["cores"][0, y_min:y_max + 1, x_min:x_max + 1]
        core_t0 = (cores_t0 != 0).astype(np.uint8)

    timestamp = valid_time.strftime("%Y-%m-%d %H:%M")

    try:
        initiations = get_initiations_at_time(timestamp)
    except FileNotFoundError as e:
        print(str(e), flush=True)
        continue

    if initiations.empty:
        print("Empty initiation", flush=True)
        continue

    ci_points = np.column_stack([
        initiations["latN_c"].values,
        initiations["lonE_c"].values,
    ])

    dist, flat_idx = tree.query(ci_points)
    j_ci, i_ci = np.unravel_index(flat_idx, (Ny, Nx))

    labels = []
    scores = []

    for k in range(len(j_ci)):
        y, s = evaluate_ci_event(
            j_ci[k],
            i_ci[k],
            lats,
            lons,
            core_t0,
            gt,
            nowcast,
            R_km=RADIUS,
        )
        labels.append(y)
        scores.append(s)

    labels = np.array(labels)
    scores = np.array(scores)

    print(
        valid_time,
        "CI:", len(initiations),
        "scores:", np.sum(~np.isnan(scores)),
        flush=True,
    )

    valid = ~np.isnan(scores)
    labels = labels[valid]
    scores = scores[valid]

    if len(labels) == 0:
        continue

    all_labels.append(labels)
    all_scores.append(scores)

if len(all_labels) == 0:
    raise RuntimeError("No valid CI events found after filtering (all lists empty).")

all_labels = np.concatenate(all_labels)
all_scores = np.concatenate(all_scores)

if len(np.unique(all_labels)) < 2:
    raise RuntimeError("Only one class present in labels after filtering; AUC is undefined.")

auc = roc_auc_score(all_labels, all_scores)
print("Global AUC:", auc, flush=True)

fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

out = f"/home/users/mendrika/Object-Based-LSTMConv/outputs/initiation/corrected/roc_ci_{RADIUS}_t{LEAD_TIME:03d}min.npz"

np.savez(
    out,
    fpr=fpr,
    tpr=tpr,
    thresholds=thresholds,
)
