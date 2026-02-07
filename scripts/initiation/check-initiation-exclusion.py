import sys
import numpy as np
import pandas as pd
from pathlib import Path
from netCDF4 import Dataset

import torch
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, roc_curve

LEAD_TIME = int(sys.argv[1])
R0_MATCH = float(sys.argv[2])

R_EXCL = 100.0
V_MATCH = 40.0
R_MAX = 120.0
R_MATCH = min(R_MAX, R0_MATCH + V_MATCH * (LEAD_TIME / 60.0))

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
    a = np.sin(dlat / 2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon / 2)**2
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
    df = pd.read_csv(filepath, sep=r"\s+", skiprows=1, names=COLUMNS)
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
    timestamp = pd.to_datetime(timestamp, utc=True).floor("15min")
    filepath = BASE_PATH / f"init_{timestamp:%Y%m}_T40_vn3_30km.txt"
    df = load_month_file(filepath)
    return df[df["datetime"] == timestamp]

def evaluate_ci_event(
    j0, i0,
    lats, lons,
    core_t0,
    core_truth,
    pancast_prob,
    R_match,
    R_excl,
    half_window=200,
):
    j1 = max(j0 - half_window, 0)
    j2 = min(j0 + half_window + 1, lats.shape[0])
    i1 = max(i0 - half_window, 0)
    i2 = min(i0 + half_window + 1, lats.shape[1])

    dist = haversine_km(
        lats[j0, i0],
        lons[j0, i0],
        lats[j1:j2, i1:i2],
        lons[j1:j2, i1:i2],
    )

    if np.any(core_t0[j1:j2, i1:i2][dist <= R_excl] == 1):
        return np.nan, np.nan

    mask = dist <= R_match
    if not np.any(mask):
        return np.nan, np.nan

    y = np.any(core_truth[j1:j2, i1:i2][mask] == 1)
    s = np.nanmax(pancast_prob[j1:j2, i1:i2][mask])

    return y, s

def load_nowcast_and_gt(year, month, day, hour, minute, lead_time):
    root = f"/gws/ssde/j25b/swift/mendrika/pancast/nowcasts/t{lead_time:03d}min"
    fpath = (
        f"{root}/"
        f"{year}{month:02d}/"
        f"{hour:02d}/"
        f"pred_{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}.pt"
    )
    data = torch.load(fpath, map_location="cpu")
    return data["mean"].numpy(), data["gt"].numpy()

parent = Path(f"/gws/ssde/j25b/swift/mendrika/pancast/nowcasts/t{LEAD_TIME:03d}min")
pt_paths = list(parent.rglob("pred_*.pt"))

all_labels = []
all_scores = []

for f in pt_paths:
    date = f.stem.split("_")[1]
    time = f.stem.split("_")[2]
    year, month, day = date[:4], date[4:6], date[6:8]
    hour, minute = time[:2], time[2:4]

    try:
        nowcast, gt = load_nowcast_and_gt(year, month, day, hour, minute, LEAD_TIME)
    except FileNotFoundError:
        continue

    core_file = f"/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/{year}/{month}/{year}{month}{day}{hour}{minute}.nc"
    if not Path(core_file).exists():
        continue

    with Dataset(core_file) as ds:
        core_t0 = (ds["cores"][0, y_min:y_max+1, x_min:x_max+1] != 0).astype(np.uint8)

    initiations = get_initiations_at_time(f"{year}-{month}-{day} {hour}:{minute}")
    if initiations.empty:
        continue

    ci_points = np.column_stack([initiations["latN_c"], initiations["lonE_c"]])
    _, flat_idx = tree.query(ci_points)
    j_ci, i_ci = np.unravel_index(flat_idx, (Ny, Nx))

    for j0, i0 in zip(j_ci, i_ci):
        y, s = evaluate_ci_event(
            j0, i0,
            lats, lons,
            core_t0,
            gt,
            nowcast,
            R_MATCH,
            R_EXCL,
        )
        if not np.isnan(s):
            all_labels.append(y)
            all_scores.append(s)

all_labels = np.array(all_labels)
all_scores = np.array(all_scores)

auc = roc_auc_score(all_labels, all_scores)
fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

j = tpr - fpr
best_idx = int(np.argmax(j))
p_thr = float(thresholds[best_idx])

forecast_yes = all_scores >= p_thr

hit = np.sum((all_labels == 1) & forecast_yes)
miss = np.sum((all_labels == 1) & (~forecast_yes))
false_alarm = np.sum((all_labels == 0) & forecast_yes)
correct_negative = np.sum((all_labels == 0) & (~forecast_yes))

pod = hit / (hit + miss) if (hit + miss) > 0 else np.nan
far = false_alarm / (hit + false_alarm) if (hit + false_alarm) > 0 else np.nan
csi = hit / (hit + miss + false_alarm) if (hit + miss + false_alarm) > 0 else np.nan

out_dir = Path("/home/users/mendrika/Object-Based-LSTMConv/outputs/initiation/matching_variable")
out_dir.mkdir(parents=True, exist_ok=True)

np.savez(
    out_dir / f"ci_results_R0{int(R0_MATCH)}_R{int(round(R_MATCH))}_t{LEAD_TIME:03d}min.npz",
    lead_time_min=LEAD_TIME,
    base_match_radius_km=R0_MATCH,
    match_radius_km=R_MATCH,
    exclusion_radius_km=R_EXCL,
    v_match_kmh=V_MATCH,
    threshold=p_thr,
    auc=auc,
    hit=hit,
    miss=miss,
    false_alarm=false_alarm,
    correct_negative=correct_negative,
    pod=pod,
    far=far,
    csi=csi,
    n_total=len(all_labels),
)

pd.DataFrame(
    {
        "lead_time_min": [LEAD_TIME],
        "base_match_radius_km": [R0_MATCH],
        "match_radius_km": [R_MATCH],
        "exclusion_radius_km": [R_EXCL],
        "v_match_kmh": [V_MATCH],
        "threshold": [p_thr],
        "AUC": [auc],
        "hits": [hit],
        "misses": [miss],
        "false_alarms": [false_alarm],
        "correct_negatives": [correct_negative],
        "POD": [pod],
        "FAR": [far],
        "CSI": [csi],
        "n_total": [len(all_labels)],
    }
).to_csv(
    out_dir / f"ci_results_R0{int(R0_MATCH)}_R{int(round(R_MATCH))}_t{LEAD_TIME:03d}min.csv",
    index=False,
)
