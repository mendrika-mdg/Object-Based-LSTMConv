import os
import sys
import torch
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import label
from datetime import datetime, timedelta

sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics  # type: ignore

# --------------------------------------------------------
# DOMAIN SETTINGS
# --------------------------------------------------------
y_min, y_max = 1403, 1914
x_min, x_max = 66, 577

geodata = np.load("/gws/nopw/j04/cocoon/SSA_domain/lat_lon_2268_2080.npz")
lons = geodata["lon"][y_min:y_max+1, x_min:x_max+1]
lats = geodata["lat"][y_min:y_max+1, x_min:x_max+1]

CONTEXT_LAT_MIN = float(np.min(lats))
CONTEXT_LAT_MAX = float(np.max(lats))
CONTEXT_LON_MIN = float(np.min(lons))
CONTEXT_LON_MAX = float(np.max(lons))


# --------------------------------------------------------
# BASIC HELPERS
# --------------------------------------------------------
def prepare_core(file):
    if not os.path.exists(file):
        raise FileNotFoundError(file)
    with Dataset(file, "r") as data:
        return data.variables["cores"][0, y_min:y_max+1, x_min:x_max+1]


def update_hour(date_dict, hours_to_add, minutes_to_add):
    dt = datetime(
        int(date_dict["year"]),
        int(date_dict["month"]),
        int(date_dict["day"]),
        int(date_dict["hour"]),
        int(date_dict["minute"]),
    )
    newt = dt + timedelta(hours=hours_to_add, minutes=minutes_to_add)

    new = {
        "year": f"{newt.year:04d}",
        "month": f"{newt.month:02d}",
        "day": f"{newt.day:02d}",
        "hour": f"{newt.hour:02d}",
        "minute": f"{newt.minute:02d}",
    }
    path = (
        f"{new['year']}/{new['month']}/"
        f"{new['year']}{new['month']}{new['day']}{new['hour']}{new['minute']}.nc"
    )
    return {"time": new, "path": path}


def extract_box(matrix, y, x, box_size=3):
    half = box_size // 2
    return matrix[
        max(0, y-half) : min(y+half+1, matrix.shape[0]),
        max(0, x-half) : min(x+half+1, matrix.shape[1]),
    ]


# --------------------------------------------------------
# STORM FEATURE EXTRACTION
# --------------------------------------------------------
def create_storm_database(data_t, lats, lons):
    cores_t = data_t["cores"][0, y_min:y_max+1, x_min:x_max+1]
    if not np.any(cores_t):
        return {}

    tir_t = data_t["tir"][0, y_min:y_max+1, x_min:x_max+1]

    Pmax_lat = data_t["max_lat"][:]
    Pmax_lon = data_t["max_lon"][:]

    valid = (
        (Pmax_lon >= CONTEXT_LON_MIN) & (Pmax_lon <= CONTEXT_LON_MAX) &
        (Pmax_lat >= CONTEXT_LAT_MIN) & (Pmax_lat <= CONTEXT_LAT_MAX)
    )
    Pmax_lat = Pmax_lat[valid]
    Pmax_lon = Pmax_lon[valid]

    labeled, _ = label(cores_t != 0)
    labels = np.unique(labeled[labeled != 0])

    dict_size = {lab: np.sum(labeled == lab) * 9 for lab in labels}

    dict_extent = {}
    for lab in labels:
        mask = labeled == lab
        dict_extent[lab] = {
            "lat_min": float(np.nanmin(lats[mask])),
            "lat_max": float(np.nanmax(lats[mask])),
            "lon_min": float(np.nanmin(lons[mask])),
            "lon_max": float(np.nanmax(lons[mask])),
        }

    dict_tir = {}
    for lab in labels:
        mask = labeled == lab
        tir_core = tir_t[mask]
        yx = np.argwhere(mask)
        y, x = yx[np.argmin(tir_core)]
        dict_tir[lab] = float(np.mean(extract_box(tir_t, y, x)))

    storms = {}
    for lat, lon in zip(Pmax_lat, Pmax_lon):
        try:
            y, x = snflics.to_yx(lat, lon, lats, lons)
            if y is None or x is None:
                continue
        except:
            continue

        lab = labeled[y, x]
        if lab == 0 or lab in storms:
            continue

        ext = dict_extent[lab]
        storms[int(lab)] = {
            "lat": lat,
            "lon": lon,
            "lat_min": ext["lat_min"],
            "lat_max": ext["lat_max"],
            "lon_min": ext["lon_min"],
            "lon_max": ext["lon_max"],
            "tir": dict_tir[lab],
            "size": dict_size[lab],
            "mask": 1,
        }

    return storms


def generate_fictional_storm(latmin, latmax, lonmin, lonmax):
    lat = np.random.uniform(latmin, latmax)
    lon = np.random.uniform(lonmin, lonmax)
    return (
        "artificial",
        {
            "lat": lat, "lon": lon,
            "lat_min": lat, "lat_max": lat,
            "lon_min": lon, "lon_max": lon,
            "tir": 30.0, "size": 0.0, "mask": 0,
        },
    )


def pad_observed_storms(storm_db, nb_x0, latmin, latmax, lonmin, lonmax):
    storms = list(storm_db.items())
    if len(storms) >= nb_x0:
        storms = sorted(storms, key=lambda x: x[1]["tir"])
        return storms[:nb_x0]

    needed = nb_x0 - len(storms)
    for _ in range(needed):
        storms.append(generate_fictional_storm(latmin, latmax, lonmin, lonmax))
    return storms


# --------------------------------------------------------
# TRANSFORM TO PER-CORE FEATURE VECTOR (INCLUDES LAG)
# --------------------------------------------------------
def transform_to_array(data, time_lag):
    arr = []
    for _, e in data:
        arr.append([
            float(e["lat"]),
            float(e["lon"]),
            float(e["lat_min"]),
            float(e["lat_max"]),
            float(e["lon_min"]),
            float(e["lon_max"]),
            float(e["tir"]),
            float(e["size"]),
            int(e["mask"]),
            float(time_lag),
        ])
    return np.array(arr, dtype=np.float32)


# --------------------------------------------------------
# PROCESS ONE FILE
# --------------------------------------------------------
def process_file(file_t, nb_x0, time_lag, lats, lons):
    try:
        with Dataset(file_t, "r") as data_t:
            if data_t["max_lat"][:].size == 0:
                return None

            storm_db = create_storm_database(data_t, lats, lons)
            storms = pad_observed_storms(
                storm_db, nb_x0,
                CONTEXT_LAT_MIN, CONTEXT_LAT_MAX,
                CONTEXT_LON_MIN, CONTEXT_LON_MAX,
            )

            feats = transform_to_array(storms, time_lag)
            return torch.tensor(feats, dtype=torch.float32)

    except Exception as e:
        print("Error:", e)
        return None


# --------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------
NB_X0 = 50
YEAR = sys.argv[1]

DATA_PATH = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"
OUTPUT_FOLDER = "/work/scratch-nopw2/mendrika/OB/ncast/raw"

all_files = [
    f for f in snflics.all_files_in(DATA_PATH)
    if snflics.get_time(f)["year"] == YEAR
       and snflics.get_time(f)["month"] in ["06", "07", "08", "09"]
]
all_files.sort()

lag_minutes = [120, 90, 60, 30, 0]  # 5 lags
lead_times = [0, 1, 3, 6]

for file_t in all_files:
    time_t = snflics.get_time(file_t)

    # required inputs
    file_before = [
        DATA_PATH + update_hour(time_t, 0, -m)["path"]
        for m in lag_minutes
    ]

    # required future target files
    file_targets = [
        DATA_PATH + update_hour(time_t, h, 0)["path"]
        for h in lead_times
    ]

    if not (all(os.path.exists(f) for f in file_before)
            and all(os.path.exists(f) for f in file_targets)):
        continue

    # Load lead-time targets
    try:
        target_cores = [prepare_core(f) for f in file_targets]
    except:
        continue

    # Skip if empty
    if any((c is None) or (not np.any(c)) for c in target_cores):
        continue

    # Domain check
    with Dataset(file_t, "r") as data_t:
        Pmax_lat = data_t["max_lat"][:]
        Pmax_lon = data_t["max_lon"][:]
        valid = (
            (Pmax_lon >= CONTEXT_LON_MIN) & (Pmax_lon <= CONTEXT_LON_MAX) &
            (Pmax_lat >= CONTEXT_LAT_MIN) & (Pmax_lat <= CONTEXT_LAT_MAX)
        )
        if np.sum(valid) == 0:
            continue

    # Build lagged inputs
    lag_tensors = []
    for lag_idx, f in enumerate(file_before):
        inp = process_file(
            f, NB_X0, time_lag=lag_minutes[lag_idx],
            lats=lats, lons=lons
        )
        if inp is not None:
            lag_tensors.append(inp)

    if len(lag_tensors) != len(lag_minutes):
        continue

    # Concatenate in time
    input_tensor = torch.cat(lag_tensors, dim=0)

    # global context
    year = int(time_t["year"])
    month = int(time_t["month"])
    day = int(time_t["day"])
    hour = int(time_t["hour"])
    minute = int(time_t["minute"])

    month_angle = 2 * np.pi * (month - 1) / 12
    tod_angle = 2 * np.pi * (hour + minute / 60) / 24

    global_context = torch.tensor([
        np.sin(month_angle), np.cos(month_angle),
        np.sin(tod_angle), np.cos(tod_angle),
    ], dtype=torch.float32)

    # Save inputs
    now_id = f"{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}"
    save_input = f"{OUTPUT_FOLDER}/inputs_t0/input-{now_id}.pt"

    torch.save({
        "input_tensor": input_tensor,      # (num_lags * NB_X0, F)
        "global_context": global_context,  # (4,)
        "lags_minutes": lag_minutes,
        "nowcast_origin": now_id,
    }, save_input)

    # Save outputs
    for h, core in zip(lead_times, target_cores):
        save_target = f"{OUTPUT_FOLDER}/targets_t{h}/target-{now_id}.pt"
        torch.save({
            "data": torch.tensor(core != 0, dtype=torch.uint8),
            "lead_time": h,
            "nowcast_origin": now_id,
        }, save_target)
