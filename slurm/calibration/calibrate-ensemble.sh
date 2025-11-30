#!/bin/bash
#SBATCH --job-name=calibrate-ensemble
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/error/%j.err

set -euo pipefail

echo "======================================================"
echo " Job started on $(hostname) at $(date)"
echo "======================================================"

# environment
module load jaspy/3.11
source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

# parameters
lead_time=$1

script=/home/users/mendrika/Object-Based-LSTMConv/scripts/calibration/calibrate-ensemble-mean.py

# verify script exists
if [ ! -f "$script" ]; then
    echo "Error: Python script not found at $script"
    exit 1
fi

echo "Running ensemble calibration:"
echo " Lead time : $lead_time"
echo "======================================================"

python "$script" "$lead_time"

echo "======================================================"
echo " Job completed successfully at $(date)"
echo "======================================================"
