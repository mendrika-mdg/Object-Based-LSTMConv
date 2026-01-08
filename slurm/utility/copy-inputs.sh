#!/bin/bash

#SBATCH --job-name=copy-to-work
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=wiser-ewsa
#SBATCH --exclude=host1114
#SBATCH -o /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/error/%j.err

set -e

# Define paths
ORIGIN=$1
TARGET="/gws/ssde/j25b/swift/mendrika/pancast/raw"

echo "Creating target directories..."
mkdir -p $TARGET/inputs_t0

echo "Copying inputs_t0 to $TARGET/inputs_t0... at $(date)"
find $ORIGIN/inputs_t0 -name "*.pt" -print0 | xargs -0 -n 100 cp -t $TARGET/inputs_t0

echo "✅ All files copied successfully at $(date)"