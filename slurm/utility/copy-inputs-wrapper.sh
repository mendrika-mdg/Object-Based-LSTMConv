#!/bin/bash

for ORIGIN in \
    "/work/scratch-nopw2/mendrika/pancast/raw" \
    "/gws/nopw/j04/wiser_ewsa/mrakotomanga/pancast/raw"
do
    sbatch /home/users/mendrika/Object-Based-LSTMConv/slurm/utility/copy-inputs.sh "$ORIGIN"
done
