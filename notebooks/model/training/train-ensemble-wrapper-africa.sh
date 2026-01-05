#!/bin/bash
# wrapper to submit final ensemble training jobs (fixed LR, all lead times)

JOB_SCRIPT="/home/users/mendrika/Object-Based-LSTMConv/notebooks/model/training/train-ensemble-africa.sh"

LEAD_TIMES=(60 90)
SEEDS=(40 134 676 1998 2025)
LR=7e-5

for LEAD_TIME in "${LEAD_TIMES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Submitting job for lead_time=${LEAD_TIME}, seed=${SEED}, lr=${LR}"
        sbatch -J ens_t${LEAD_TIME}_s${SEED}_lr${LR} \
            "$JOB_SCRIPT" "$LEAD_TIME" "$SEED" "$LR"
        sleep 5
    done
done

echo "All final ensemble jobs submitted successfully."
