#!/bin/bash
#SBATCH --job-name=Reanalysis05
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=Reanalysis05.out
#SBATCH --error=Reanalysis05.err
#SBATCH -p normal
#SBATCH -c 2

module load python/3.9.0
python3 Reanalysis05.py