#!/bin/bash
#SBATCH --job-name=cgm_vqa_hf
#SBATCH --output=logs/hf_eval_%j.log
#SBATCH --error=logs/hf_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -e

mkdir -p logs
cd /nas/longleaf/home/txueying/CGM-VQA

# Create or reuse local virtual environment
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -q torch torchvision transformers accelerate pillow

# Run the evaluation script
python scripts/eval/run_eval_llava7b.py
