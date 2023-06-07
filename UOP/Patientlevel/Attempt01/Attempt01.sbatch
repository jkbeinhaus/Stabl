#!/bin/bash
#SBATCH --job-name=Attempt01
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=Attempt01.out
#SBATCH --error=Attempt01.err
#SBATCH -p normal
#SBATCH -c 2

module load python/3.9.0
python3 Attempt01.py