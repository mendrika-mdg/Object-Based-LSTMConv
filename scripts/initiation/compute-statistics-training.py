import pandas as pd
from glob import glob
from pathlib import Path
import sys

# season argument
SEASON = sys.argv[1]

# local exclusion radius in km
EXCL_RADIUS = int(sys.argv[2])

# define months
if SEASON == "JJAS":
    MONTHS = [6, 7, 8, 9]
elif SEASON == "SON":
    MONTHS = [10, 11]
elif SEASON == "DJF":
    MONTHS = [12, 1, 2]
else:
    MONTHS = [3, 4, 5]

# lead dependent guard radius in km
GUARD_BY_LEAD = {
    30: 100,
    60: 180,
    90: 250,
    120: 350,
}

# collect files
files = []
for m in MONTHS:
    pattern = (
        f"/home/users/mendrika/Object-Based-LSTMConv/outputs/initiation/training/"
        f"ci_core_assoc/ci_core_assoc_*{m:02d}_t*min.csv"
    )
    files.extend(glob(pattern))

files = sorted(files)

if len(files) == 0:
    raise ValueError("No files found")

# load data
dfs = [pd.read_csv(f, parse_dates=["valid_time"]) for f in files]
df = pd.concat(dfs, ignore_index=True)

print("Total CI samples:", len(df), flush=True)

# inclusion radii of interest
R_IN_LIST = [25, 40]

results = []

# seasonal statistics
for lead in sorted(df["lead_time_min"].unique()):

    df_lead = df[df["lead_time_min"] == lead]

    guard_km = GUARD_BY_LEAD[lead]

    # local exclusion
    local_clean = df_lead[f"excl_{EXCL_RADIUS:03d}km"] == 0

    # guard exclusion
    guard_clean = df_lead[f"excl_{guard_km:03d}km"] == 0

    # combined causal mask
    mask_clean = local_clean & guard_clean

    for Rin in R_IN_LIST:

        col = f"incl_{Rin:03d}km"

        results.append({
            "season": SEASON,
            "lead_time_min": lead,
            "local_excl_km": EXCL_RADIUS,
            "guard_excl_km": guard_km,
            "incl_radius_km": Rin,
            "p_incl_causal": df_lead.loc[mask_clean, col].mean(),
            "n_causal": mask_clean.sum()
        })

df_stats = pd.DataFrame(results)

results_diurnal = []

# diurnal statistics
for lead in sorted(df["lead_time_min"].unique()):

    guard_km = GUARD_BY_LEAD[lead]

    for hour in range(24):

        df_subset = df[
            (df["lead_time_min"] == lead) &
            (df["hour"] == hour)
        ]

        if len(df_subset) == 0:
            continue

        # local exclusion
        local_clean = df_subset[f"excl_{EXCL_RADIUS:03d}km"] == 0

        # guard exclusion
        guard_clean = df_subset[f"excl_{guard_km:03d}km"] == 0

        # combined causal mask
        mask_clean = local_clean & guard_clean

        for Rin in R_IN_LIST:

            col = f"incl_{Rin:03d}km"

            results_diurnal.append({
                "season": SEASON,
                "lead_time_min": lead,
                "hour": hour,
                "local_excl_km": EXCL_RADIUS,
                "guard_excl_km": guard_km,
                "incl_radius_km": Rin,
                "p_incl_causal": df_subset.loc[mask_clean, col].mean(),
                "n_causal": mask_clean.sum()
            })

df_diurnal = pd.DataFrame(results_diurnal)

# save outputs
out_dir = Path("/home/users/mendrika/Object-Based-LSTMConv/outputs/initiation/training/ci_core_assoc_stats")
out_dir.mkdir(parents=True, exist_ok=True)

file_seasonal = out_dir / f"ci_core_stats_{SEASON}_causal_excl{EXCL_RADIUS:03d}.csv"
df_stats.to_csv(file_seasonal, index=False)

file_diurnal = out_dir / f"ci_core_stats_{SEASON}_diurnal_causal_excl{EXCL_RADIUS:03d}.csv"
df_diurnal.to_csv(file_diurnal, index=False)

print("Saved seasonal:", file_seasonal, flush=True)
print("Saved diurnal:", file_diurnal, flush=True)
