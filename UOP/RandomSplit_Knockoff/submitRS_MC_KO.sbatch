#!/bin/bash
#SBATCH --job-name=RS_MC_KO
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=RS_MC_KO.out
#SBATCH --error=RS_MC_KO.err
#SBATCH -p normal
#SBATCH -c 2

module load python/3.9.0
python3 RS_MC_KO.py