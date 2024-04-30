#!/bin/bash
# --- this job will be run on any available node
# To run it, use `sbatch slurm_markov.sh`
#SBATCH --job-name="Markov Password Generation"
#SBATCH --error="slurm_markov.err"
#SBATCH --output="slurm_markov.output"

python3 driver.py --load-markov markov.pickle -r 0 --guesses_count 10_000_000