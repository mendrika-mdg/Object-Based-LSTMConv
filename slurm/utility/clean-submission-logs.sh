#!/bin/bash

set -e

clean_dir () {
    dir="$1"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        rm -rf "$dir"/*
        echo "Cleaned: $dir"
    else
        echo "Skipped (empty or missing): $dir"
    fi
}

clean_dir /home/users/mendrika/Object-Based-LSTMConv/slurm/submission/error
clean_dir /home/users/mendrika/Object-Based-LSTMConv/slurm/submission/output
clean_dir /home/users/mendrika/Object-Based-LSTMConv/wandb

clean_dir /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/error
clean_dir /home/users/mendrika/Object-Based-LSTMConv/slurm/submission-logs/output
