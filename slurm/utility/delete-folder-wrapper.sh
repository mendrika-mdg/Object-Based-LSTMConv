#!/bin/bash
# Launcher for raw data preparation jobs across multiple years

for LEAD_TIME in 0 1 2 3 4 5 6; do 
    sbatch -J "del-lead${LEAD_TIME}" /home/users/mendrika/Object-Based-LSTMConv/slurm/utility/delete-folder.sh $LEAD_TIME
done