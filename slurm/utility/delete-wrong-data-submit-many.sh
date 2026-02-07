#!/bin/bash

# Submit deletion for inputs
sbatch -J "input" \
    /home/users/mendrika/Object-Based-LSTMConv/slurm/utility/delete-wrong-data.sh \
    /work/scratch-nopw2/mendrika/pancast/raw/inputs_t0

# Submit deletion for all target lead times
for i in 030 060 090 120; do
    sbatch -J "targets_t${i}min" \
        /home/users/mendrika/Object-Based-LSTMConv/slurm/utility/delete-wrong-data.sh \
        "/work/scratch-nopw2/mendrika/pancast/raw/targets_t${i}min"
done
