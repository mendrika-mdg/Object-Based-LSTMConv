import os
import sys
import torch
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import label
from datetime import datetime, timedelta

sys.path.insert(1, "/home/users/mendrika/SSA/SA/module")
import snflics # type: ignore

def load_geodata():
    geodata = np.load(
        "/work/scratch-nopw2/mendrika/lat_lon_2268_2080.npz",
        mmap_mode="r"
    )
    return geodata["lat"], geodata["lon"]


lats, lons = load_geodata()

print(lats.shape)

YEAR = sys.argv[1]

print(YEAR)