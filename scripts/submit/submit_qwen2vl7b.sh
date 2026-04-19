#!/bin/bash
#SBATCH --job-name=cgmvqa_qwen2vl7b
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/qwen2vl7b_%j.out
#SBATCH --error=logs/qwen2vl7b_%j.err

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
VENV=$WORKFS/hf_venv_latest
if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    mkdir -p $WORKFS
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Install packages (fast on rerun if already cached)
pip install --upgrade pip setuptools wheel -q
pip install -q \
    torch==2.4.0+cu118 torchvision==0.19.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -U -q transformers accelerate pillow qwen-vl-utils protobuf tokenizers

# ── HuggingFace cache → /work (large space) ───────────────────────────────────
export HF_HOME=$WORKFS/hf_cache
export TRANSFORMERS_CACHE=$WORKFS/hf_cache/hub
mkdir -p "$HF_HOME"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"
which python3
python3 --version

# Use offline mode — model already cached in HF_HOME, avoids 429 rate-limit on HF API
# export TRANSFORMERS_OFFLINE=1
# export HF_HUB_OFFLINE=1

python3 scripts/eval/run_eval_qwen2vl7b_hf.py

echo "Job finished: $(date)"
