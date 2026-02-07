#!/bin/bash
# launcher for CI–PANCAST evaluation

JOB_SCRIPT="/home/users/mendrika/Object-Based-LSTMConv/slurm/initiation/check-initiation.sh"

# lead times (minutes)
LEAD_TIMES=("030" "060" "090" "120")

# radii (km)
RADII=("30" "50")

for LEAD_TIME in "${LEAD_TIMES[@]}"; do
    for RADIUS in "${RADII[@]}"; do
        echo "Submitting job for lead_time=${LEAD_TIME}, radius=${RADIUS} km..."
        sbatch -J "t${LEAD_TIME}_R${RADIUS}" "$JOB_SCRIPT" "$LEAD_TIME" "$RADIUS"
        sleep 3
    done
done

echo "All jobs submitted successfully."
