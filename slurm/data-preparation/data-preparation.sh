#!/bin/bash
#SBATCH --job-name=data-preparation
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/Object-Based-LSTMConv/slurm/submission/output/%j.out
#SBATCH -e /home/users/mendrika/Object-Based-LSTMConv/slurm/submission/error/%j.err

set -e

module load jaspy/3.11
source /home/users/mendrika/SSA/bin/activate

export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

year=$1
script=/home/users/mendrika/Object-Based-LSTMConv/scripts/data-preparation/raw-data-preparation.py

if [ ! -f "$script" ]; then
    echo "Error: Python script not found at $script"
    exit 1
fi

python "$script" "$year"

echo "Job completed successfully."
