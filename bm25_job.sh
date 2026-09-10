#!/bin/bash
#SBATCH --job-name=bm25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=bm25_%j.out
#SBATCH --error=bm25_%j.err

cd /home2/sachirao/Assignment1

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3.6 run_bm25.py --dataset all --split val --max-impressions -1
