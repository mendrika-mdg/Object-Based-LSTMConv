#!/bin/bash
#SBATCH --job-name=apply-scaler
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/error/%j.err

set -e

# load env
module load jaspy/3.11
source /home/users/mendrika/SSA/bin/activate

# disable HDF5 file locking
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# read args
partition=$1
lead_time=$2

# path to python script
script=/home/users/mendrika/Object-Based-LSTMConv/scripts/data-preparation/apply-scaler.py

# check script exists
if [ ! -f "$script" ]; then
    echo "Error: Python script not found at $script"
    exit 1
fi

# run
python "$script" "$partition" "$lead_time"

echo "Job completed successfully."
