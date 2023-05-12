#!/bin/bash
#SBATCH --job-name=UOP_SS_LOPOKO
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=UOP_SS_LOPOKO.out
#SBATCH --error=UOP_SS_LOPOKO.err
#SBATCH -p normal
#SBATCH -c 2

module load python/3.9.0
python3 UOPStabilitySelectionLOPOKO.py