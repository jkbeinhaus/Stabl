#!/bin/bash
#SBATCH --job-name=Reanalysis05Val
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=Reanalysis05Val.out
#SBATCH --error=Reanalysis05Val.err
#SBATCH -p normal
#SBATCH -c 2

module load python/3.9.0
python3 Reanalysis05Val.py