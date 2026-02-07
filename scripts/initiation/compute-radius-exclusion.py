import pandas as pd
from glob import glob
from pathlib import Path

# load all CI-core association files
files = sorted(glob(
    "/home/users/mendrika/Object-Based-LSTMConv/outputs/initiation/training/"
    "ci_core_assoc/ci_core_assoc_*_t*min.csv"
))

if len(files) == 0:
    raise ValueError("No files found")

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

print("Total CI samples:", len(df), flush=True)


# exclusion radii
R_EXCL_LIST = [10, 15, 20, 25, 30, 40, 50, 60]

rows = []

for Rex in R_EXCL_LIST:
    clean_fraction = (df[f"excl_{Rex:03d}km"] == 0).mean()
    rows.append({
        "R_excl_km": Rex,
        "clean_fraction": clean_fraction
    })

df_excl = pd.DataFrame(rows)

print(df_excl, flush=True)


# save
out_dir = Path("/home/users/mendrika/Object-Based-LSTMConv/outputs/initiation/radius_selection")
out_dir.mkdir(parents=True, exist_ok=True)

df_excl.to_csv(out_dir / "exclusion_selection_all_data.csv", index=False)

print("Saved exclusion table.", flush=True)
