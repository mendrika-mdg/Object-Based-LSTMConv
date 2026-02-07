import pandas as pd
from glob import glob
from pathlib import Path


# chosen exclusion radius
Rex_choice = 30

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


# restrict to clean CI
df_clean = df[df[f"excl_{Rex_choice:03d}km"] == 0]

print("Clean CI samples:", len(df_clean), flush=True)


rows = []

for lead in sorted(df_clean["lead_time_min"].unique()):

    tmp = df_clean[df_clean["lead_time_min"] == lead]

    rows.append({
        "lead_time_min": lead,
        "n_clean": len(tmp),
        "median_km": tmp["dmin_t_km"].median(),
        "q75_km": tmp["dmin_t_km"].quantile(0.75),
        "q90_km": tmp["dmin_t_km"].quantile(0.90)
    })

df_quantiles = pd.DataFrame(rows)

print(df_quantiles, flush=True)


# save
out_dir = Path("/home/users/mendrika/Object-Based-LSTMConv/outputs/initiation/radius_selection")
out_dir.mkdir(parents=True, exist_ok=True)

df_quantiles.to_csv(out_dir / "inclusion_selection_all_data.csv", index=False)

print("Saved inclusion table.", flush=True)
