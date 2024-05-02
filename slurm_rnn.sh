#!/bin/bash
# --- this job will be run on any available node
# To run it, use `sbatch slurm_rnn.sh`
#SBATCH --job-name="RNN Password Generation"
#SBATCH --error="slurm_rnn.err"
#SBATCH --output="slurm_rnn.output"

python3 driver.py --load-rnn saved.pickle -r 0 --guesses_count 10_000_000