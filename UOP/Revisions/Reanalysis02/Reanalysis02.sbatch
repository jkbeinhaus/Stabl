#!/bin/bash
#SBATCH --job-name=Reanalysis02CV
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=Reanalysis02CV.out
#SBATCH --error=Reanalysis02CV.err
#SBATCH -p normal
#SBATCH -c 2

module load python/3.9.0
python3 Reanalysis02CV.py