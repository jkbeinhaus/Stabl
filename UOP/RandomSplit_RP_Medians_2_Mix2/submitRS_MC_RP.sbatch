#!/bin/bash
#SBATCH --job-name=RS_MC_RP_new
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=RS_MC_RP.out
#SBATCH --error=RS_MC_RP.err
#SBATCH -p normal
#SBATCH -c 2

module load python/3.9.0
python3 RS_MC_RP.py