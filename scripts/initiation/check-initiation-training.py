import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from netCDF4 import Dataset
from datetime import datetime, timedelta
from scipy.spatial import cKDTree


# Path containing CI initiation text files
INIT_PATH = Path("/gws/nopw/j04/wiser_ewsa/mrakotomanga/initiations/txt_op")

# Column names for CI initiation files
COLUMNS = [
    "mo", "dy", "hr", "mn",
    "latN_c", "lonE_c",
    "i_c", "j_c", "i", "j",
    "topo_sd", "wf", "minT", "Rmax"
]


# Load monthly CI file and construct datetime
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
        utc=True,
    )
    return df


# Update timestamp by adding minutes and build new core file path
def update_time(date_dict, minutes_to_add):
    time_obj = datetime(
        int(date_dict["year"]),
        int(date_dict["month"]),
        int(date_dict["day"]),
        int(date_dict["hour"]),
        int(date_dict["minute"])
    )
    updated = time_obj + timedelta(minutes=minutes_to_add)
    new_date_dict = {
        "year": f"{updated.year:04d}",
        "month": f"{updated.month:02d}",
        "day": f"{updated.day:02d}",
        "hour": f"{updated.hour:02d}",
        "minute": f"{updated.minute:02d}",
    }
    file_path = (
        f"{new_date_dict['year']}/{new_date_dict['month']}/"
        f"{new_date_dict['year']}{new_date_dict['month']}"
        f"{new_date_dict['day']}{new_date_dict['hour']}{new_date_dict['minute']}.nc"
    )
    return {"time": new_date_dict, "path": file_path}


# Load core data and convert to binary mask
def prepare_core_binary(file, y_min, y_max, x_min, x_max):
    with Dataset(file, "r") as data:
        cores = data.variables["cores"][0, y_min:y_max + 1, x_min:x_max + 1]
    return (cores != 0).astype(np.uint8)


# Compute great-circle distance in km
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# Read command-line arguments
LEAD_TIME = int(sys.argv[1])
YEAR = sys.argv[2]
MONTH = sys.argv[3]


# Load MSG latitude and longitude grids
geodata = np.load("/gws/nopw/j04/cocoon/SSA_domain/lat_lon_2268_2080.npz")
y_min, y_max = 48, 2062
x_min, x_max = 81, 2267

lons = geodata["lon"][y_min:y_max + 1, x_min:x_max + 1]
lats = geodata["lat"][y_min:y_max + 1, x_min:x_max + 1]

Ny, Nx = lats.shape


# Fill value for invalid grid cells
VALID_FILL = -999.999

# Mask valid grid points
valid_mask = (lats != VALID_FILL) & (lons != VALID_FILL)

# Row and column indices of valid grid cells
jj, ii = np.where(valid_mask)

# Coordinates used to build KD-tree
latlon_points = np.column_stack([lats[jj, ii], lons[jj, ii]])

# KD-tree for nearest neighbour snapping
tree = cKDTree(latlon_points)

# Inclusion radii in km
R_IN_LIST = [10, 15, 20, 25, 30, 40, 50, 60]

# Extended exclusion radii in km for guard filtering
R_EXCL_LIST = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 250, 300, 350]


# Root directory for core NetCDF files
DATA_PATH = "/gws/nopw/j04/cocoon/SSA_domain/ch9_wavelet/"


# Load CI initiation file
init_file = INIT_PATH / f"init_{YEAR}{MONTH}_T40_vn3_30km.txt"
if not init_file.exists():
    raise FileNotFoundError(init_file)

df_ci = load_month_file(init_file)
print("Total CI in file:", len(df_ci), flush=True)


rows = []


# Pixel size in km
PIXEL_SIZE_KM = 3.0

# Maximum radius used
R_MAX = max(max(R_IN_LIST), max(R_EXCL_LIST))

# Half window size in pixels to cover maximum radius
half_window = int(np.ceil(R_MAX / PIXEL_SIZE_KM)) + 2


# Loop over CI grouped by timestamp
for timestamp, df_t in df_ci.groupby("datetime"):

    time_dict = {
        "year": f"{timestamp.year:04d}",
        "month": f"{timestamp.month:02d}",
        "day": f"{timestamp.day:02d}",
        "hour": f"{timestamp.hour:02d}",
        "minute": f"{timestamp.minute:02d}",
    }

    # Core file at t0
    file_t0 = os.path.join(
        DATA_PATH,
        f"{time_dict['year']}/{time_dict['month']}/"
        f"{time_dict['year']}{time_dict['month']}"
        f"{time_dict['day']}{time_dict['hour']}{time_dict['minute']}.nc"
    )

    # Core file at t0 + lead time
    file_t = os.path.join(
        DATA_PATH,
        update_time(time_dict, minutes_to_add=LEAD_TIME)["path"]
    )

    if not os.path.exists(file_t0) or not os.path.exists(file_t):
        continue

    try:
        core_t0 = prepare_core_binary(file_t0, y_min, y_max, x_min, x_max)
        core_t = prepare_core_binary(file_t, y_min, y_max, x_min, x_max)
    except OSError:
        continue

    # CI lat/lon coordinates
    ci_points = np.column_stack([
        df_t["latN_c"].values,
        df_t["lonE_c"].values
    ])

    # Snap CI to nearest grid cell
    _, idx = tree.query(ci_points)
    j_ci = jj[idx]
    i_ci = ii[idx]

    for j0, i0 in zip(j_ci.astype(int), i_ci.astype(int)):

        # Local window indices
        j1 = max(j0 - half_window, 0)
        j2 = min(j0 + half_window + 1, Ny)
        i1 = max(i0 - half_window, 0)
        i2 = min(i0 + half_window + 1, Nx)

        lat_win = lats[j1:j2, i1:i2]
        lon_win = lons[j1:j2, i1:i2]

        # Distance from CI to all grid points in window
        dist = haversine_km(lats[j0, i0], lons[j0, i0], lat_win, lon_win)

        core0_win = core_t0[j1:j2, i1:i2]
        coret_win = core_t[j1:j2, i1:i2]

        dmin_t0 = float(np.min(dist[core0_win == 1])) if np.any(core0_win == 1) else np.nan
        dmin_t = float(np.min(dist[coret_win == 1])) if np.any(coret_win == 1) else np.nan

        row = {
            "valid_time": timestamp,
            "lead_time_min": LEAD_TIME,
            "month": timestamp.month,
            "hour": timestamp.hour,
            "dmin_t0_km": dmin_t0,
            "dmin_t_km": dmin_t,
        }

        # Exclusion flags at t0
        for Rex in R_EXCL_LIST:
            row[f"excl_{Rex:03d}km"] = int(np.any(core0_win[dist <= Rex] == 1))

        # Inclusion flags at t0 + lead time
        for Rin in R_IN_LIST:
            row[f"incl_{Rin:03d}km"] = int(np.any(coret_win[dist <= Rin] == 1))

        rows.append(row)


df_out = pd.DataFrame(rows)

out_dir = Path("/home/users/mendrika/Object-Based-LSTMConv/outputs/initiation/training/ci_core_assoc")
out_dir.mkdir(parents=True, exist_ok=True)

out_csv = out_dir / f"ci_core_assoc_{YEAR}{MONTH}_t{LEAD_TIME:03d}min.csv"
df_out.to_csv(out_csv, index=False)

print("Saved:", out_csv, flush=True)
print("N CI samples:", len(df_out), flush=True)
print("half_window pixels:", half_window, flush=True)
