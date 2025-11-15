#!/bin/bash

#SBATCH --job-name=nbx0-ws
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/Object-Based-LSTMConv/slurm/output/%j.out
#SBATCH -e /home/users/mendrika/Object-Based-LSTMConv/slurm/error/%j.err

# Exit immediately if any command fails
set -e

# Load Python environment
module load jaspy/3.11
source /home/users/mendrika/SSA/bin/activate

# Define region parameters
domain_lat_min=2.1
domain_lat_max=22.5
domain_lon_min=-21.5
domain_lon_max=-1.45
region_name="ws"

# Run Python script with arguments
python /home/users/mendrika/Object-Based-LSTMConv/scripts/wa-nb-x0.py \
  "$domain_lat_min" "$domain_lat_max" "$domain_lon_min" "$domain_lon_max" "$region_name"

echo "Job completed successfully."