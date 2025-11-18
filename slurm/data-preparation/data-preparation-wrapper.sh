#!/bin/bash
# Launcher for raw data preparation jobs across multiple years

JOB_SCRIPT="/home/users/mendrika/Object-Based-LSTMConv/slurm/data-preparation/data-preparation.sh"

for year in $(seq 2004 2024); do
    echo "Submitting job for year ${year}..."
    sbatch -J "${year}" "${JOB_SCRIPT}" "${year}"
    sleep 1
done

echo "All jobs submitted successfully."
