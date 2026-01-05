#!/bin/bash
#SBATCH --job-name=delete
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --account=wiser-ewsa
#SBATCH --exclude=host1114
#SBATCH -o /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/nb-x0/output/%j.out
#SBATCH -e /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/log/submission-history/nb-x0/error/%j.err

LEAD_TIME=$1

rm -r /gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Pancast-v1-512/t${LEAD_TIME}