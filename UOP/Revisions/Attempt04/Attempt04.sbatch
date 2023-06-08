#!/bin/bash
#SBATCH --job-name=Attempt04
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=Attempt04.out
#SBATCH --error=Attempt04.err
#SBATCH -p normal
#SBATCH -c 2

module load python/3.9.0
python3 Attempt04.py