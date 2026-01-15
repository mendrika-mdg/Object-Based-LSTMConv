#!/bin/bash
#SBATCH --job-name=ens_t${1}_s${2}_lr${3}
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --exclude=gpuhost006,gpuhost015
#SBATCH -o /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/output/%j.out
#SBATCH -e /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/error/%j.err

echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs allocated: ${CUDA_VISIBLE_DEVICES}"
nvidia-smi

source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONHASHSEED=0
export PL_TORCH_DISTRIBUTED_BACKEND=gloo

lead_time=$1
seed=$2
lr=$3

if [ -z "$lead_time" ] || [ -z "$seed" ] || [ -z "$lr" ]; then
    echo "Usage: sbatch ensemble.sh <lead_time> <seed> <learning_rate>"
    exit 1
fi

echo "Starting distributed training for lead_time=${lead_time}, seed=${seed}, lr=${lr}"

torchrun --standalone --nproc_per_node=4 \
    /home/users/mendrika/Object-Based-LSTMConv/notebooks/model/training/pancast_64_sharp_decoder.py \
    "$lead_time" "$seed" "$lr"

echo "Training completed at $(date)"
