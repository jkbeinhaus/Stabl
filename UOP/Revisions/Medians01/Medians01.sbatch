#!/bin/bash
#SBATCH --job-name=Medians01
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=Medians01.out
#SBATCH --error=Medians01.err
#SBATCH -p normal
#SBATCH -c 2

module load python/3.9.0
python3 Medians01.py