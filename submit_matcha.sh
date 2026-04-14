#!/bin/bash
#SBATCH --job-name=cgmvqa_matcha
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/matcha_%j.out
#SBATCH --error=logs/matcha_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load python/3.12.4
module load cuda/11.8

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR=/nas/longleaf/home/$USER/CGM-VQA
WORKFS=/work/users/t/x/$USER        # Longleaf /work filesystem (high-perf)
cd $WORKDIR
mkdir -p logs results

# ── Virtual environment on /work (large space, not home quota) ────────────────
VENV=$WORKFS/hf_venv
if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    mkdir -p $WORKFS
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Install packages (fast on rerun if already cached)
pip install --upgrade pip setuptools wheel -q
# Preserve existing torch version to not break flash-attn for Qwen2/LLaVA
pip install -q transformers==4.40.0 accelerate pillow protobuf sentencepiece

# ── HuggingFace cache → /work (large space) ───────────────────────────────────
export HF_HOME=$WORKFS/hf_cache
export TRANSFORMERS_CACHE=$WORKFS/hf_cache/hub
mkdir -p "$HF_HOME"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"
which python3
python3 --version

# Run the python evaluation script
python3 run_eval_matcha_hf.py

echo "Job finished: $(date)"
