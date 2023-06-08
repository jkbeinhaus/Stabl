#!/bin/bash
#SBATCH --job-name=Attempt03
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=Attempt03.out
#SBATCH --error=Attempt03.err
#SBATCH -p normal
#SBATCH -c 2

module load python/3.9.0
python3 Attempt03.py