#!/bin/bash
#SBATCH --job-name=COVID19
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=COVID19.out
#SBATCH --error=COVID19.err
#SBATCH -p normal
#SBATCH -c 2

module load python/3.9.0
python3 COVID19Benchmarks.py