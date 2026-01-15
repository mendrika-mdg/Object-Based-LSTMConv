import os
import sys
import torch
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import label
from datetime import datetime, timedelta

sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics # type: ignore


def get_time(file_path):
    """
    Extract zero-padded time components from a file path like:
    /.../2025/02/05/1345/Convective_struct_extended_202502051345_000.nc

    Returns:
        dict with keys: 'year', 'month', 'day', 'hour', 'minute'
    """
    basename = os.path.basename(file_path)

    # Extract the datetime string from the filename
    parts = basename.split("_")
    
    timestamp = parts[-2]
    if len(timestamp) != 12:
        raise ValueError(f"Invalid timestamp format in filename: {timestamp}")

    return {
        "year":   timestamp[0:4],
        "month":  timestamp[4:6],
        "day":    timestamp[6:8],
        "hour":   timestamp[8:10],
        "minute": timestamp[10:12]
    }


def load_geodata():
    geodata = Dataset(
        "/gws/ssde/j25b/swift/rt_cores/geoloc_grids/nxny2268_2080_nxnyds164580_blobdx0.04491576_arean41_n27_27_79.nc",
        mmap_mode="r"
    )
    return geodata["lats_mid"], geodata["lons_mid"]

# For a given region, add yx bounds and context domain
y_min, y_max = 48, 2062
x_min, x_max = 76, 2262

# Import geodata and crop it accordingly
lats, lons = load_geodata()

lats = lats[y_min:y_max+1, x_min:x_max+1]
lons = lons[y_min:y_max+1, x_min:x_max+1]

lats = np.where(lats == -999.999, np.nan, lats)
lons = np.where(lons == -999.999, np.nan, lons)

CONTEXT_LAT_MIN = -35
CONTEXT_LAT_MAX = 24
CONTEXT_LON_MIN = -18
CONTEXT_LON_MAX = 51

def update_hour(date_dict, hours_to_add, minutes_to_add):
    """
    Add hours to a datetime dictionary and return the updated dict and a generated file path.

    Args:
        date_dict     (dict): Keys: 'year', 'month', 'day', 'hour', 'minute' as strings, e.g. "01", "23"
        hours_to_add   (int): Number of hours to add.
        minutes_to_add (int): Number of minutes to add.

    Returns:
        dict: {
            'time': updated time dictionary with zero-padded strings,
            'path': file path in format /.../YYYYMMDDHHMM_000.nc
        }
    """
    # Parse original time
    time_obj = datetime(
        int(date_dict["year"]),
        int(date_dict["month"]),
        int(date_dict["day"]),
        int(date_dict["hour"]),
        int(date_dict["minute"])
    )

    # Add hours and minutes
    updated = time_obj + timedelta(hours=hours_to_add, minutes=minutes_to_add)

    # Create updated dictionary with padded strings
    new_date_dict = {
        "year":   f"{updated.year:04d}",
        "month":  f"{updated.month:02d}",
        "day":    f"{updated.day:02d}",
        "hour":   f"{updated.hour:02d}",
        "minute": f"{updated.minute:02d}"
    }

    # Build file path safely using single quotes inside f-strings
    path_core = f"/gws/ssde/j25b/swift/rt_cores/{new_date_dict['year']}/{new_date_dict['month']}/{new_date_dict['day']}/{new_date_dict['hour']}{new_date_dict['minute']}"
    file_path = f"{path_core}/Convective_struct_extended_{new_date_dict['year']}{new_date_dict['month']}{new_date_dict['day']}{new_date_dict['hour']}{new_date_dict['minute']}_000.nc"

    return {'time': new_date_dict, 'path': file_path}

def extract_box(matrix, y, x, box_size=3):
    half = box_size // 2
    y_min_box = max(y - half, 0)
    y_max_box = min(y + half + 1, matrix.shape[0])
    x_min_box = max(x - half, 0)
    x_max_box = min(x + half + 1, matrix.shape[1])
    return matrix[y_min_box:y_max_box, x_min_box:x_max_box]


def create_storm_database(data_t, lats, lons):
    # Identify storm cores and extract features for each core

    tir_t = data_t["cores"][y_min:y_max+1, x_min:x_max+1].data
    temp_t = tir_t < 0
    Pmax_lat, Pmax_lon = data_t["Pmax_lat"][:], data_t["Pmax_lon"][:]

    valid = (
        (Pmax_lon >= CONTEXT_LON_MIN) & (Pmax_lon <= CONTEXT_LON_MAX) &
        (Pmax_lat >= CONTEXT_LAT_MIN) & (Pmax_lat <= CONTEXT_LAT_MAX)
    )
    Pmax_lat, Pmax_lon = Pmax_lat[valid], Pmax_lon[valid]

    labeled_array, _ = label(temp_t)
    core_labels = np.unique(labeled_array[labeled_array != 0])

    dict_storm_size = {lab: np.sum(labeled_array == lab) * 9 for lab in core_labels}

    dict_storm_extent = {}
    for lab in core_labels:
        mask = labeled_array == lab

        lat_vals = lats[mask]
        lon_vals = lons[mask]

        if np.all(np.isnan(lat_vals)) or np.all(np.isnan(lon_vals)):
            continue

        dict_storm_extent[lab] = {
            "lat_min": float(np.nanmin(lat_vals)),
            "lat_max": float(np.nanmax(lat_vals)),
            "lon_min": float(np.nanmin(lon_vals)),
            "lon_max": float(np.nanmax(lon_vals)),
        }


    dict_storm_temperature = {}
    for lab in core_labels:
        mask = labeled_array == lab
        tir_core = tir_t[mask]
        yx_indices = np.argwhere(mask)
        y, x = yx_indices[np.argmin(tir_core)]
        box = extract_box(tir_t, y, x)
        dict_storm_temperature[lab] = float(np.nanmean(box))

    storm_database = {}
    for lat, lon in zip(Pmax_lat, Pmax_lon):
        try:
            y_idx, x_idx = snflics.to_yx(lat, lon, lats, lons)
            if y_idx is None or x_idx is None:
                continue
        except (IndexError, TypeError):
            continue
        lab = labeled_array[y_idx, x_idx]
        if lab == 0 or lab in storm_database:
            continue

        ext = dict_storm_extent[lab]
        storm_database[int(lab)] = {
            "lat": lat,
            "lon": lon,
            "lat_min": ext["lat_min"],
            "lat_max": ext["lat_max"],
            "lon_min": ext["lon_min"],
            "lon_max": ext["lon_max"],
            "tir": dict_storm_temperature[lab],
            "size": dict_storm_size[lab],
            "mask": 1
        }
    return storm_database


def generate_fictional_storm(context_lat_min, context_lat_max,
                             context_lon_min, context_lon_max):
    # Generate a dummy non-convective storm entry with mask=0
    lat = np.random.uniform(context_lat_min, context_lat_max)
    lon = np.random.uniform(context_lon_min, context_lon_max)

    storm = {
        "lat": lat,
        "lon": lon,
        "lat_min": lat,
        "lat_max": lat,
        "lon_min": lon,
        "lon_max": lon,
        "tir": 30.0,
        "size": 0.0,
        "mask": 0
    }

    return ("artificial", storm)

def pad_observed_storms(storm_db, nb_x0,
                        context_lat_min, context_lat_max,
                        context_lon_min, context_lon_max):
    # Ensure a fixed number of storm cores by truncating or padding

    storm_list = list(storm_db.items())

    if len(storm_list) >= nb_x0:
        sorted_db = sorted(storm_list, key=lambda item: item[1]["tir"])
        return sorted_db[:nb_x0]

    needed = nb_x0 - len(storm_list)
    for _ in range(needed):
        storm_list.append(
            generate_fictional_storm(
                context_lat_min=context_lat_min,
                context_lat_max=context_lat_max,
                context_lon_min=context_lon_min,
                context_lon_max=context_lon_max
            )
        )

    return storm_list


def transform_to_array(data):
    # Transform list of storms into an array of local per-core features

    result = []
    for _, entry in data:
        lat = float(entry["lat"])
        lon = float(entry["lon"])
        lat_min = float(entry.get("lat_min", lat))
        lat_max = float(entry.get("lat_max", lat))
        lon_min = float(entry.get("lon_min", lon))
        lon_max = float(entry.get("lon_max", lon))
        tir = float(entry["tir"])
        size = float(entry["size"])
        mask = int(entry["mask"])

        # [lat, lon, lat_min, lat_max, lon_min, lon_max, tir, size, mask]
        result.append([
            lat, lon,
            lat_min, lat_max,
            lon_min, lon_max,
            tir, size,
            mask
        ])

    return np.array(result, dtype=np.float32)



def process_file(file_t, nb_x0,
                 lats, lons,
                 CONTEXT_LAT_MIN, CONTEXT_LAT_MAX,
                 CONTEXT_LON_MIN, CONTEXT_LON_MAX):

    try:
        with Dataset(file_t, "r") as data_t:

            x0_lat = data_t["Pmax_lat"][:]
            x0_lon = data_t["Pmax_lon"][:]

            if x0_lat.size == 0 or x0_lon.size == 0:
                return None

            storm_database = create_storm_database(data_t, lats, lons)

            X_features = pad_observed_storms(
                storm_database, nb_x0,
                CONTEXT_LAT_MIN, CONTEXT_LAT_MAX,
                CONTEXT_LON_MIN, CONTEXT_LON_MAX
            )

            input_features = transform_to_array(X_features)

            if np.isnan(input_features).any():
                print(f"NaNs in features for {file_t}")
                return None

            input_tensor = torch.tensor(input_features, dtype=torch.float32)

        return input_tensor

    except Exception as e:
        print(f"Error processing {file_t}: {e}")
        return None


NB_X0 = 100

YEAR = sys.argv[1]

DATA_PATH = "/gws/ssde/j25b/swift/rt_cores"
YEAR_PATH = os.path.join(DATA_PATH, YEAR)

from glob import glob
print(f"Scanning {YEAR_PATH}", flush=True)

all_files = sorted(glob(os.path.join(YEAR_PATH, "**", "*.nc"), recursive=True))
print("YEAR =", YEAR, flush=True)
print("Total files found:", len(all_files), flush=True)

GWS_OUTPUT_DIR = "/gws/ssde/j25b/swift/mendrika/pancast/raw/2026/inputs_t0"
os.makedirs(GWS_OUTPUT_DIR, exist_ok=True)

lag_before_t = [120, 90, 60, 30, 0]

for file_t in all_files:

    time_t = get_time(file_t)

    file_before_t = [
        update_hour(time_t, hours_to_add=0, minutes_to_add=-m)["path"]
        for m in lag_before_t
    ]

    year = int(time_t["year"])
    month = int(time_t["month"])
    day = int(time_t["day"])
    hour = int(time_t["hour"])
    minute = int(time_t["minute"])

    NOWCAST_ORIGIN = f"{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}"
    out_path = os.path.join(GWS_OUTPUT_DIR, f"input-{NOWCAST_ORIGIN}.pt")

    if os.path.exists(out_path):
        continue

    if not all(os.path.exists(f) for f in file_before_t):
        print(f"Missing required lag files for {NOWCAST_ORIGIN}", flush=True)
        continue

    file_t0 = file_before_t[-1]

    try:
        with Dataset(file_t0, "r") as data_t:
            Pmax_lat = data_t["Pmax_lat"][:]
            Pmax_lon = data_t["Pmax_lon"][:]

            valid = (
                (Pmax_lon >= CONTEXT_LON_MIN) & (Pmax_lon <= CONTEXT_LON_MAX) &
                (Pmax_lat >= CONTEXT_LAT_MIN) & (Pmax_lat <= CONTEXT_LAT_MAX)
            )

            if np.count_nonzero(valid) == 0:
                print(f"No core in the domain for {NOWCAST_ORIGIN}", flush=True)
                continue

    except Exception as e:
        print(f"Failed reading t0 file for {NOWCAST_ORIGIN}: {e}", flush=True)
        continue

    lag_tensors = []
    ok = True

    for f in file_before_t:

        t_tensor_local = process_file(
            f,
            nb_x0=NB_X0,
            lats=lats,
            lons=lons,
            CONTEXT_LAT_MIN=CONTEXT_LAT_MIN,
            CONTEXT_LAT_MAX=CONTEXT_LAT_MAX,
            CONTEXT_LON_MIN=CONTEXT_LON_MIN,
            CONTEXT_LON_MAX=CONTEXT_LON_MAX
        )

        if t_tensor_local is None:
            ok = False
            break

        t_time = get_time(f)
        t_month = int(t_time["month"])
        t_hour = int(t_time["hour"])
        t_minute = int(t_time["minute"])

        month_angle = 2 * np.pi * (t_month - 1) / 12.0
        tod_angle = 2 * np.pi * (t_hour + t_minute / 60.0) / 24.0

        time_features = torch.tensor(
            [
                np.sin(month_angle), np.cos(month_angle),
                np.sin(tod_angle), np.cos(tod_angle)
            ],
            dtype=torch.float32
        ).unsqueeze(0).repeat(NB_X0, 1)

        t_tensor_full = torch.cat([time_features, t_tensor_local], dim=1)
        lag_tensors.append(t_tensor_full)

    if not ok:
        print(f"Incomplete lag sequence for {NOWCAST_ORIGIN}", flush=True)
        continue

    input_tensor = torch.stack(lag_tensors, dim=0)

    if input_tensor.shape != (len(lag_before_t), NB_X0, 13):
        print(f"Bad tensor shape for {NOWCAST_ORIGIN}: {tuple(input_tensor.shape)}", flush=True)
        continue

    torch.save(
        {
            "input_tensor": input_tensor,
            "nowcast_origin": NOWCAST_ORIGIN,
            "lags_minutes": lag_before_t,
        },
        out_path
    )

    print(f"Saved input tensor: {out_path}", flush=True)
