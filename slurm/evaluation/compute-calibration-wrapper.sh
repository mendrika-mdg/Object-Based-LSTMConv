#!/bin/bash
# Submit FSS computation jobs for all lead times and hours

JOB_SCRIPT="/home/users/mendrika/Object-Based-LSTMConv/slurm/evaluation/compute-calibration.sh"

# Lead times to process
LEAD_TIMES=("1" "3" "6")

for LEAD_TIME in "${LEAD_TIMES[@]}"; do
        echo "Submitting job for lead_time=${LEAD_TIME}"
        sbatch -J "calibration_t${LEAD_TIME}h" "$JOB_SCRIPT" "$LEAD_TIME"
        sleep 1  # small delay to avoid submission bursts
    done

echo "All jobs submitted successfully."
