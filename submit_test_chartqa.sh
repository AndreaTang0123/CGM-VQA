#!/bin/bash
#SBATCH --job-name=yesno_test
#SBATCH --partition=general
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/yesno_test_%j.out
#SBATCH --error=logs/yesno_test_%j.err

module purge
module load python/3.12.4
module load cuda/11.8

WORKFS=/work/users/t/x/$USER
source $WORKFS/hf_venv/bin/activate
pip install -q transformers==4.40.0

export HF_HOME=$WORKFS/hf_cache
export TRANSFORMERS_CACHE=$WORKFS/hf_cache/hub

python3 -u run_test_yesno_seq.py
