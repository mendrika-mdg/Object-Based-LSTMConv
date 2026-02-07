#!/bin/bash

JOB_SCRIPT="/home/users/mendrika/Object-Based-LSTMConv/slurm/initiation/check-statistics-training.sh"

# seasons
SEASONS=("JJAS" "SON" "DJF" "MAM")

# fixed local exclusion radius
EXCL_RADIUS="30"

for SEASON in "${SEASONS[@]}"; do
    echo "Submitting job for season=${SEASON}, R_excl=${EXCL_RADIUS} km"
    sbatch -J "stats_${SEASON}_R${EXCL_RADIUS}" \
        "$JOB_SCRIPT" "$SEASON" "$EXCL_RADIUS"
    sleep 2
done

echo "All jobs submitted successfully."
