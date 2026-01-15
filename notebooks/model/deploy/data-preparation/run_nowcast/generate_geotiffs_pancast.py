import sys, csv, os, datetime
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
from rasterio.transform import from_origin
import subprocess

root = "/gws/ssde/j25b/swift/UKCEH_nowcast_portal"
userid = "mendrika"
modelname = "pancast"

LATLON_MISSING = -999.999
DATA_MISSING = 0.0

lats = np.load(
    "/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/geolocation/nrt_lats_africa.npy"
)
lons = np.load(
    "/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/deploy/geolocation/nrt_lons_africa.npy"
)

assert lats.shape == lons.shape == (2015, 2186)

TO_GEOTIFF = "/work/scratch-nopw2/mendrika/pancast-live/log/ready_geotiff.csv"
NOWCASTS_FOLDER = "/work/scratch-nopw2/mendrika/pancast-live"
PROCESSED_FILES = "/work/scratch-nopw2/mendrika/pancast-live/log/geotiff/processed_times.csv"
MISSED_FILES = "/work/scratch-nopw2/mendrika/pancast-live/log/geotiff/missed_times.csv"

LEAD_TIMES_MIN = [30, 60, 90, 120]


def is_already_processed(time_dict):
    if not Path(PROCESSED_FILES).exists():
        return False
    with open(PROCESSED_FILES, newline="") as f:
        reader = csv.DictReader(f)
        return any(all(row[k] == time_dict[k] for k in time_dict) for row in reader)


def append_time(time_dict, log_path):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists() or path.stat().st_size == 0

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["year", "month", "day", "hour", "minute"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow(time_dict)


def align_prediction_to_nrt(pred, lats, lons):
    ny, nx = lats.shape

    if pred.shape == (ny, nx):
        return pred

    if pred.shape == (ny, nx + 1):
        return pred[:, :nx]

    raise ValueError(
        f"Incompatible PANCAST shape {pred.shape} "
        f"for NRT grid {lats.shape}"
    )

def write_geotiff(out_file, data, lats, lons):
    ny, nx = data.shape

    data = np.where(np.isnan(data), DATA_MISSING, data).astype(np.float32)

    valid = (lats != LATLON_MISSING) & (lons != LATLON_MISSING)

    lat_min = lats[valid].min()
    lat_max = lats[valid].max()
    lon_min = lons[valid].min()
    lon_max = lons[valid].max()

    dy = (lat_max - lat_min) / (ny - 1)
    dx = (lon_max - lon_min) / (nx - 1)

    transform = from_origin(lon_min, lat_max, dx, dy)

    with rasterio.open(
        out_file,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        nodata=DATA_MISSING,
        transform=transform,
        compress="DEFLATE",
        tiled=True,
    ) as dst:
        dst.write(np.flipud(data), 1)


def warp_to_3857(in_file):
    tmp_file = in_file.replace(".tif", "_tmp.tif")

    subprocess.run(
        [
            "gdalwarp",
            "-s_srs", "EPSG:4326",
            "-t_srs", "EPSG:3857",
            "-r", "bilinear",
            "-overwrite",
            in_file,
            tmp_file,
        ],
        check=True,
    )

    os.replace(tmp_file, in_file)


if __name__ == "__main__":

    df = pd.read_csv(TO_GEOTIFF)

    year, month, day, hour, minute = df.loc[0][
        ["year", "month", "day", "hour", "minute"]
    ]

    time_dict = {
        "year": str(year),
        "month": f"{int(month):02d}",
        "day": f"{int(day):02d}",
        "hour": f"{int(hour):02d}",
        "minute": f"{int(minute):02d}",
    }

    if is_already_processed(time_dict):
        sys.exit(0)

    nowcast_origin = datetime.datetime(
        int(year), int(month), int(day), int(hour), int(minute)
    )

    out_dir = os.path.join(
        root, f"{userid}_{modelname}", nowcast_origin.strftime("%Y%m%d")
    )
    os.makedirs(out_dir, exist_ok=True)

    for lt in LEAD_TIMES_MIN:
        fpath = (
            f"{NOWCASTS_FOLDER}/nowcasts_t{lt:03d}/"
            f"nowcast_t{lt:03d}_from_{nowcast_origin.strftime('%Y%m%d_%H%M')}.npy"
        )

        if not Path(fpath).exists():
            append_time(time_dict, MISSED_FILES)
            sys.exit(1)

        pred_raw = np.load(fpath)
        pred = align_prediction_to_nrt(pred_raw, lats, lons)

        assert pred.shape == lats.shape, (
            f"Final mismatch: pred {pred.shape} vs grid {lats.shape}"
        )

        out_file = os.path.join(
            out_dir,
            f"nowcast_{nowcast_origin.strftime('%Y%m%d%H%M_')}{lt:04d}.tif",
        )
 
        write_geotiff(out_file, pred, lats, lons)
        warp_to_3857(out_file)


    append_time(time_dict, PROCESSED_FILES)
