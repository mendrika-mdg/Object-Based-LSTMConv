#!/bin/bash
set -euo pipefail

JOB_SCRIPT="/home/users/mendrika/Object-Based-LSTMConv/slurm/scale/apply-scaler.sh"

if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Job script not found: $JOB_SCRIPT"
    exit 1
fi

PARTITIONS=("train" "val")

for PARTITION in "${PARTITIONS[@]}"; do
    echo "Submitting job for partition=${PARTITION}"
    sbatch -J "scale_${PARTITION}" "$JOB_SCRIPT" "$PARTITION"
    sleep 1
done

echo "All jobs submitted successfully."
