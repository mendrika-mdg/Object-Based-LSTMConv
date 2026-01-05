#!/bin/bash
#SBATCH --job-name=shard-train
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/error/%j.err

set -e

# load environment
module load jaspy/3.11
source /home/users/mendrika/SSA/bin/activate

export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

# arguments
partition=$1        # must be "train"
lead_time=$2        # e.g. 030
year_start=$3       # e.g. 2004
year_end=$4         # e.g. 2007

if [ -z "$partition" ] || [ -z "$lead_time" ] || [ -z "$year_start" ] || [ -z "$year_end" ]; then
    echo "Usage: sbatch shard-train.slurm train <lead_time> <year_start> <year_end>"
    exit 1
fi

script=/home/users/mendrika/Object-Based-LSTMConv/scripts/shards/create-shards-africa-train.py

if [ ! -f "$script" ]; then
    echo "Error: Python script not found at $script"
    exit 1
fi

echo "Running train shard creation:"
echo "  partition  = $partition"
echo "  lead time  = $lead_time min"
echo "  years      = $year_start–$year_end"

python "$script" "$partition" "$lead_time" "$year_start" "$year_end"

echo "Job completed successfully."
