#!/bin/bash
#SBATCH --job-name=shard-creation
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/error/%j.err

set -e

# load environment
module load jaspy/3.11
source /home/users/mendrika/SSA/bin/activate

# disable file locking for HDF5
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# read arguments
partition=$1
lead_time=$2

# path to python script
script=/home/users/mendrika/Object-Based-LSTMConv/scripts/shards/create-shards.py

# check script exists
if [ ! -f "$script" ]; then
    echo "Error: Python script not found at $script"
    exit 1
fi

# run shard creation
python "$script" "$partition" "$lead_time"

echo "Job completed successfully."
