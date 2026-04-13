#!/bin/bash
#SBATCH --job-name=cgmvqa_minicpmv4
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/minicpmv4_%j.out
#SBATCH --error=logs/minicpmv4_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load python/3.12.4
module load cuda/11.8

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR=/nas/longleaf/home/$USER/CGM-VQA
WORKFS=/work/users/t/x/$USER
cd $WORKDIR
mkdir -p logs results

# ── Virtual environment on /work ──────────────────────────────────────────────
VENV=$WORKFS/hf_venv
if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    mkdir -p $WORKFS
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Install packages
pip install --upgrade pip setuptools wheel -q
pip install -q \
    torch==2.4.0+cu118 torchvision==0.19.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers accelerate pillow einops timm sentencepiece

# ── HuggingFace cache → /work ─────────────────────────────────────────────────
export HF_HOME=$WORKFS/hf_cache
export TRANSFORMERS_CACHE=$WORKFS/hf_cache/hub
mkdir -p "$HF_HOME"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"
which python3
python3 --version

python3 run_eval_minicpmv4_hf.py

echo "Job finished: $(date)"
