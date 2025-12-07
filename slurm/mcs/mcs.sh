#!/bin/bash

#SBATCH --job-name=count-mcs
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/Object-Based-LSTMConv/slurm/submission/output/%j.out
#SBATCH -e /home/users/mendrika/Object-Based-LSTMConv/slurm/submission/error/%j.err

# Exit immediately if any command fails
set -e

# Load Python environment
module load jaspy/3.11
source /home/users/mendrika/SSA/bin/activate

# Run Python script with arguments
python /home/users/mendrika/Object-Based-LSTMConv/scripts/mcs/count.py

echo "Job completed successfully."