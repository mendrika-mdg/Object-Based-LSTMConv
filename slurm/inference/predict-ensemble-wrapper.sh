#!/bin/bash
# launcher for ensemble prediction

JOB_SCRIPT="/home/users/mendrika/Object-Based-LSTMConv/slurm/inference/predict-ensemble.sh"

# configuration
YEARS=("2019")
MONTHS=("06" "07" "08" "09")
HOURS=($(seq -w 0 23))
LEAD_TIMES=("1" "3" "6")

for YEAR in "${YEARS[@]}"; do
  for MONTH in "${MONTHS[@]}"; do
    for HOUR in "${HOURS[@]}"; do
      for LEAD_TIME in "${LEAD_TIMES[@]}"; do
        JOB_NAME="oblstm${LEAD_TIME}_${YEAR}${MONTH}_${HOUR}"
        echo "Submitting job: ${JOB_NAME}"
        sbatch -J "${JOB_NAME}" \
               "$JOB_SCRIPT" "$LEAD_TIME" "$YEAR" "$MONTH" "$HOUR"
        sleep 0.5
      done
    done
  done
done

echo "All ensemble nowcast jobs submitted successfully."
