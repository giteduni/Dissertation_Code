#!/bin/bash
#$ -N fastopic_best
#$ -pe sharedmem 8
#$ -l h_rt=08:00:00
#$ -l h_vmem=16G
#$ -j y
#$ -o /exports/eddie/scratch/s2829951/fastopic/best_run.log
#$ -wd /exports/eddie/scratch/s2829951/fastopic

set -e
PROJ=/exports/eddie/scratch/s2829951/fastopic

. /etc/profile.d/modules.sh
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /exports/eddie/scratch/s2829951/.conda/envs/fastopic

export HF_HOME=/exports/eddie/scratch/s2829951/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

cd "$PROJ"
python "$PROJ/fastopic_best.py" \
    --input "$PROJ/hustle_core.csv" \
    --num-topics 10 --seeds 5 --out-dir "$PROJ/out_best"
