#!/bin/bash
#SBATCH --job-name=Reanalysis04Val
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=Reanalysis04Val.out
#SBATCH --error=Reanalysis04Val.err
#SBATCH -p normal
#SBATCH -c 2

module load python/3.9.0
python3 Reanalysis04Val.py