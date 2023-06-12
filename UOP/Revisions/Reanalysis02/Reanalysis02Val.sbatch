#!/bin/bash
#SBATCH --job-name=Reanalysis02Val
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=Reanalysis02Val.out
#SBATCH --error=Reanalysis02Val.err
#SBATCH -p normal
#SBATCH -c 2

module load python/3.9.0
python3 Reanalysis02Val.py