#!/bin/bash
#SBATCH --job-name=UOP_MC_RP
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=UOP_MC_RP.out
#SBATCH --error=UOP_MC_RP.err
#SBATCH -p normal
#SBATCH -c 2

module load python/3.9.0
python3 UOP_MC_RP.py