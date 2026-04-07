#!/bin/bash
#SBATCH --job-name=cgmvqa_llava7b
#SBATCH --partition=l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/llava7b_%j.out
#SBATCH --error=logs/llava7b_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=txueying@unc.edu

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load python/3.11.9
module load cuda/12.1

# Activate your conda/venv environment (edit path as needed)
source ~/.bashrc
conda activate cgmvqa          # or: source /path/to/venv/bin/activate

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR=/nas/longleaf/home/$USER/CGM-VQA-1   # ← edit to your actual path
cd $WORKDIR

# Create logs dir if not present
mkdir -p logs results

# Cache HuggingFace models to scratch to avoid quota issues
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export HF_HOME=/scratch/$USER/hf_cache

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python3 run_eval_llava7b_hf.py --batch-size 4

echo "Job finished: $(date)"
