#!/bin/bash
# --- this job will be run on any available node
# To run it, use `sbatch slurm_rnn.sh`
#SBATCH --job-name="RNN Password Training"
#SBATCH --error="slurm_rnn.err"
#SBATCH --output="slurm_rnn.output"

python3 final_rnn.py