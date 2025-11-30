#!/bin/bash
# launcher for ensemble prediction

JOB_SCRIPT="/home/users/mendrika/Object-Based-LSTMConv/slurm/calibration/calibrate-ensemble.sh"

LEAD_TIMES=("1" "3" "6")

for LEAD_TIME in "${LEAD_TIMES[@]}"; do
  JOB_NAME="model${LEAD_TIME}"
  echo "Submitting job: ${JOB_NAME}"
  sbatch -J "${JOB_NAME}" \
          "$JOB_SCRIPT" "$LEAD_TIME"
  sleep 0.5
done


echo "All ensemble calibration jobs submitted successfully."