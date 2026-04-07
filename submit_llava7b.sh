#!/bin/bash
#SBATCH --job-name=cgmvqa_llava7b
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/llava7b_%j.out
#SBATCH --error=logs/llava7b_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load python/3.11.9
module load cuda/12.1

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR=/nas/longleaf/home/$USER/CGM-VQA
cd $WORKDIR

# Create logs dir if not present
mkdir -p logs results

# Create or activate virtual environment
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel -q
python -m pip install -q torch torchvision transformers accelerate pillow

# Cache HuggingFace models to scratch to avoid quota issues
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export HF_HOME=/scratch/$USER/hf_cache

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"
which python3

python3 run_eval_llava7b_hf.py

echo "Job finished: $(date)"
