import os
import sys
import torch
import numpy as np

from glob import glob
from netCDF4 import Dataset
from scipy.ndimage import label
from datetime import datetime, timedelta

sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics


def get_time(file_path):
    basename = os.path.basename(file_path)
    parts = basename.split("_")
    timestamp = parts[-2]

    if len(timestamp) != 12:
        raise ValueError(f"Invalid timestamp format in filename: {timestamp}")

    return {
        "year": timestamp[0:4],
        "month": timestamp[4:6],
        "day": timestamp[6:8],
        "hour": timestamp[8:10],
        "minute": timestamp[10:12],
    }


def load_geodata():
    with Dataset(
        "/gws/ssde/j25b/swift/rt_cores/geoloc_grids/nxny2268_2080_nxnyds164580_blobdx0.04491576_arean41_n27_27_79.nc",
        mmap_mode="r",
    ) as ds:
        lats = np.asarray(ds["lats_mid"][:])
        lons = np.asarray(ds["lons_mid"][:])
    return lats, lons



def update_hour(date_dict, hours_to_add, minutes_to_add):
    time_obj = datetime(
        int(date_dict["year"]),
        int(date_dict["month"]),
        int(date_dict["day"]),
        int(date_dict["hour"]),
        int(date_dict["minute"]),
    )

    updated = time_obj + timedelta(hours=hours_to_add, minutes=minutes_to_add)

    new_date_dict = {
        "year": f"{updated.year:04d}",
        "month": f"{updated.month:02d}",
        "day": f"{updated.day:02d}",
        "hour": f"{updated.hour:02d}",
        "minute": f"{updated.minute:02d}",
    }

    path_core = (
        f"/gws/ssde/j25b/swift/rt_cores/"
        f"{new_date_dict['year']}/{new_date_dict['month']}/{new_date_dict['day']}/"
        f"{new_date_dict['hour']}{new_date_dict['minute']}"
    )

    file_path = (
        f"{path_core}/Convective_struct_extended_"
        f"{new_date_dict['year']}{new_date_dict['month']}{new_date_dict['day']}"
        f"{new_date_dict['hour']}{new_date_dict['minute']}_000.nc"
    )

    return {"time": new_date_dict, "path": file_path}


def extract_box(matrix, y, x, box_size=3):
    half = box_size // 2
    y0 = max(y - half, 0)
    y1 = min(y + half + 1, matrix.shape[0])
    x0 = max(x - half, 0)
    x1 = min(x + half + 1, matrix.shape[1])
    return matrix[y0:y1, x0:x1]


def create_storm_database(
    data_t,
    lats_crop,
    lons_crop,
    context_lat_min,
    context_lat_max,
    context_lon_min,
    context_lon_max,
):
    tir_t = np.asarray(data_t["cores"][y_min:y_max+1, x_min:x_max+1])

    # cold cloud mask
    temp_t = tir_t < 0

    labeled_array, _ = label(temp_t)
    core_labels = np.unique(labeled_array[labeled_array != 0])

    # storm size (km²)
    dict_storm_size = {
        int(lab): float(np.sum(labeled_array == lab) * 9.0)
        for lab in core_labels
    }

    # extents
    dict_storm_extent = {}
    for lab in core_labels:
        mask = labeled_array == lab

        lat_vals = lats_crop[mask]
        lon_vals = lons_crop[mask]

        if lat_vals.size == 0:
            continue

        dict_storm_extent[int(lab)] = {
            "lat_min": float(np.nanmin(lat_vals)),
            "lat_max": float(np.nanmax(lat_vals)),
            "lon_min": float(np.nanmin(lon_vals)),
            "lon_max": float(np.nanmax(lon_vals)),
        }

    # mean temperature around coldest pixel
    dict_storm_temperature = {}
    for lab in core_labels:
        mask = labeled_array == lab
        tir_core = tir_t[mask]

        if tir_core.size == 0:
            continue

        yx = np.argwhere(mask)
        y, x = yx[np.argmin(tir_core)]

        box = extract_box(tir_t, y, x)
        dict_storm_temperature[int(lab)] = float(np.nanmean(box))

    # build database using coldest-pixel location as Pmax
    storm_database = {}

    for lab in core_labels:
        mask = labeled_array == lab
        tir_core = tir_t[mask]

        if tir_core.size == 0:
            continue

        yx = np.argwhere(mask)

        # coldest pixel (Pmax equivalent)
        y, x = yx[np.argmin(tir_core)]

        lat = float(lats_crop[y, x])
        lon = float(lons_crop[y, x])

        # domain filter
        if not (
            context_lat_min <= lat <= context_lat_max
            and context_lon_min <= lon <= context_lon_max
        ):
            continue

        ext = dict_storm_extent.get(int(lab))
        if ext is None:
            continue

        storm_database[int(lab)] = {
            "lat": lat,
            "lon": lon,
            "lat_min": ext["lat_min"],
            "lat_max": ext["lat_max"],
            "lon_min": ext["lon_min"],
            "lon_max": ext["lon_max"],
            "tir": dict_storm_temperature.get(int(lab), float("nan")),
            "size": dict_storm_size.get(int(lab), 0.0),
            "mask": 1,
        }

    return storm_database


def generate_fictional_storm(context_lat_min, context_lat_max,
                             context_lon_min, context_lon_max):
    lat = float(np.random.uniform(context_lat_min, context_lat_max))
    lon = float(np.random.uniform(context_lon_min, context_lon_max))

    storm = {
        "lat": lat,
        "lon": lon,
        "lat_min": lat,
        "lat_max": lat,
        "lon_min": lon,
        "lon_max": lon,
        "tir": 30.0,
        "size": 0.0,
        "mask": 0,
    }

    return ("artificial", storm)


def pad_observed_storms(storm_db, nb_x0,
                        context_lat_min, context_lat_max,
                        context_lon_min, context_lon_max):
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
                context_lon_max=context_lon_max,
            )
        )

    return storm_list


def transform_to_array(data):
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

        result.append([lat, lon, lat_min, lat_max, lon_min, lon_max, tir, size, mask])

    return np.array(result, dtype=np.float32)


def process_file(
    file_t,
    nb_x0,
    lats_crop,
    lons_crop,
    context_lat_min,
    context_lat_max,
    context_lon_min,
    context_lon_max,
):
    try:
        with Dataset(file_t, "r") as data_t:
            t_time = get_time(file_t)
            t_month = int(t_time["month"])
            t_hour = int(t_time["hour"])
            t_minute = int(t_time["minute"])

            month_angle = 2 * np.pi * (t_month - 1) / 12.0
            tod_angle = 2 * np.pi * (t_hour + t_minute / 60.0) / 24.0

            time_features = torch.tensor(
                [np.sin(month_angle), np.cos(month_angle), np.sin(tod_angle), np.cos(tod_angle)],
                dtype=torch.float32,
            ).unsqueeze(0).repeat(nb_x0, 1)

            storm_database = create_storm_database(
                data_t,
                lats_crop=lats_crop,
                lons_crop=lons_crop,
                context_lat_min=context_lat_min,
                context_lat_max=context_lat_max,
                context_lon_min=context_lon_min,
                context_lon_max=context_lon_max,
            )

            if len(storm_database) == 0:
                return None

            X_features = pad_observed_storms(
                storm_database,
                nb_x0,
                context_lat_min,
                context_lat_max,
                context_lon_min,
                context_lon_max,
            )

            core_features_np = transform_to_array(X_features)

            if np.isnan(core_features_np).any():
                print(f"NaNs in features for {file_t}", flush=True)
                return None

            core_features = torch.from_numpy(core_features_np)
            input_tensor = torch.cat([time_features, core_features], dim=1)
            return input_tensor

    except Exception as e:
        print(f"Error processing {file_t}: {e}", flush=True)
        return None


def load_target_core_mask(file_path, y_min, y_max, x_min, x_max):
    with Dataset(file_path, "r") as ds:
        cores = np.asarray(ds["cores"][y_min:y_max + 1, x_min:x_max + 1])
    return cores < 0


NB_X0 = 100

YEAR = sys.argv[1]
MONTH = sys.argv[2]

DATA_PATH = "/gws/ssde/j25b/swift/rt_cores"
YEAR_PATH = os.path.join(DATA_PATH, YEAR, MONTH)

print(f"Scanning {YEAR_PATH} for month {MONTH}", flush=True)

all_files = sorted(glob(os.path.join(YEAR_PATH, "**", "*.nc"), recursive=True))
print("YEAR =", YEAR, flush=True)
print("Total files found:", len(all_files), flush=True)

if  len(all_files) ==0:
    sys.exit("No data found")

OUTPUT_FOLDER = "/gws/ssde/j25b/swift/mendrika/pancast/test"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

lag_before_t = [120, 90, 60, 30, 0]
lead_times = [30, 60, 90, 120]

CONTEXT_LAT_MIN = -35
CONTEXT_LAT_MAX = 24
CONTEXT_LON_MIN = -18
CONTEXT_LON_MAX = 51

y_min, y_max = 48, 2062
x_min, x_max = 77, 2262

lats_full, lons_full = load_geodata()
lats_crop = np.asarray(lats_full[y_min:y_max + 1, x_min:x_max + 1])
lons_crop = np.asarray(lons_full[y_min:y_max + 1, x_min:x_max + 1])

lats_crop = np.where(lats_crop == -999.999, np.nan, lats_crop)
lons_crop = np.where(lons_crop == -999.999, np.nan, lons_crop)

for file_t in all_files:
    time_t = get_time(file_t)

    file_before_t = [
        update_hour(time_t, hours_to_add=0, minutes_to_add=-m)["path"]
        for m in lag_before_t
    ]

    if not all(os.path.exists(f) for f in file_before_t):
        print(f"Missing required lag files for base file {file_t}", flush=True)
        continue

    file_lead_times = [
        update_hour(time_t, hours_to_add=0, minutes_to_add=h)["path"]
        for h in lead_times
    ]

    if not all(os.path.exists(f) for f in file_lead_times):
        print(f"Missing required lead-time files for base file {file_t}", flush=True)
        continue

    year = int(time_t["year"])
    month = int(time_t["month"])
    day = int(time_t["day"])
    hour = int(time_t["hour"])
    minute = int(time_t["minute"])
    NOWCAST_ORIGIN = f"{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}"

    INPUT_LT0 = f"{OUTPUT_FOLDER}/inputs_t0/input-{NOWCAST_ORIGIN}.pt"
    os.makedirs(os.path.dirname(INPUT_LT0), exist_ok=True)

    OUTPUT_PATHS = {
        f"LT{m:03d}min": f"{OUTPUT_FOLDER}/targets_t{m:03d}min/target-{NOWCAST_ORIGIN}.pt"
        for m in lead_times
    }
    for path in OUTPUT_PATHS.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

    file_t0 = file_before_t[-1]

    try:
        with Dataset(file_t0, "r") as data_t0:
            storm_db0 = create_storm_database(
                data_t0,
                lats_crop=lats_crop,
                lons_crop=lons_crop,
                context_lat_min=CONTEXT_LAT_MIN,
                context_lat_max=CONTEXT_LAT_MAX,
                context_lon_min=CONTEXT_LON_MIN,
                context_lon_max=CONTEXT_LON_MAX,
            )

            if len(storm_db0) == 0:
                print(f"No core in the domain for {NOWCAST_ORIGIN}", flush=True)
                continue

    except Exception as e:
        print(f"Failed reading t0 file for {NOWCAST_ORIGIN}: {e}", flush=True)
        continue


    lag_tensors = []
    ok = True

    for f in file_before_t:
        t_tensor = process_file(
            f,
            nb_x0=NB_X0,
            lats_crop=lats_crop,
            lons_crop=lons_crop,
            context_lat_min=CONTEXT_LAT_MIN,
            context_lat_max=CONTEXT_LAT_MAX,
            context_lon_min=CONTEXT_LON_MIN,
            context_lon_max=CONTEXT_LON_MAX,
        )

        if t_tensor is None:
            ok = False
            break

        lag_tensors.append(t_tensor)

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
        INPUT_LT0,
    )

    for l in lead_times:
        try:
            target_mask = load_target_core_mask(
                update_hour(time_t, hours_to_add=0, minutes_to_add=l)["path"],
                y_min=y_min, y_max=y_max, x_min=x_min, x_max=x_max,
            )
        except Exception as e:
            print(f"Skipping targets for {NOWCAST_ORIGIN} at {l} min: {e}", flush=True)
            continue

        target_tensor = torch.tensor(target_mask, dtype=torch.uint8)
        output_file_path = OUTPUT_PATHS[f"LT{l:03d}min"]

        torch.save(
            {
                "data": target_tensor,
                "lead_time_minutes": l,
                "nowcast_origin": NOWCAST_ORIGIN,
            },
            output_file_path,
        )

    print(f"Saved input tensor: {INPUT_LT0}", flush=True)
