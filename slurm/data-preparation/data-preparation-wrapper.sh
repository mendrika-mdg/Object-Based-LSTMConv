#!/bin/bash
# Launcher for raw data preparation jobs across multiple years and months

JOB_SCRIPT="/home/users/mendrika/Object-Based-LSTMConv/slurm/data-preparation/data-preparation.sh"

for year in 2024; do
    for month in $(seq -w 10 12); do
        echo "Submitting job for ${year}-${month}..."
        sbatch -J "${year}_${month}" "${JOB_SCRIPT}" "${year}" "${month}"
        sleep 2
    done
done

echo "All jobs submitted successfully."
