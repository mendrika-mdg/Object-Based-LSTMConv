#!/bin/bash

#SBATCH --job-name=R_inc
#SBATCH --time=24:00:00
#SBATCH --mem=124G
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/Object-Based-LSTMConv/slurm/submission/output/%j.out
#SBATCH -e /home/users/mendrika/Object-Based-LSTMConv/slurm/submission/error/%j.err

set -e

module load jaspy/3.11
source /home/users/mendrika/SSA/bin/activate

python /home/users/mendrika/Object-Based-LSTMConv/scripts/initiation/compute-radius-inclusion.py

echo "Job completed successfully."
