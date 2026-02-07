#!/bin/bash
# launcher for ensemble prediction

set -euo pipefail

JOB_SCRIPT="/home/users/mendrika/Object-Based-LSTMConv/slurm/inference/predict-ensemble-africa.sh"

# configuration
YEARS=("2024")
MONTHS=($(seq -w 1 12))
HOURS=($(seq -w 0 23))
LEAD_TIMES=("030" "060" "090" "120")

for YEAR in "${YEARS[@]}"; do
  for MONTH in "${MONTHS[@]}"; do
    for HOUR in "${HOURS[@]}"; do
      for LEAD_TIME in "${LEAD_TIMES[@]}"; do

        JOB_NAME="oblstm${LEAD_TIME}_${YEAR}${MONTH}_${HOUR}"
        echo "Submitting job: ${JOB_NAME}"

        sbatch -J "${JOB_NAME}" \
               "${JOB_SCRIPT}" "${LEAD_TIME}" "${YEAR}" "${MONTH}" "${HOUR}"

        sleep 0.5
      done
    done
  done
done

echo "All ensemble nowcast jobs submitted successfully."
