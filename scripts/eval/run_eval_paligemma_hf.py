"""
Batch evaluation of CGM-VQA using PaliGemma (google/paligemma-3b-mix-448) via HuggingFace.

Requirements:
    pip install transformers accelerate pillow torch

Results saved to: results/paligemma_hf_results.json

Usage:
    python3 run_eval_paligemma_hf.py
    python3 run_eval_paligemma_hf.py --batch-size 1 --model google/paligemma-3b-mix-448
"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent.parent
IMAGE_BASE   = BASE_DIR / "graphs_cropped"
EVAL_FILE    = BASE_DIR / "metadata" / "eval_questions.json"
RESULTS_DIR  = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "paligemma_hf_results.json"

DEFAULT_MODEL      = "google/paligemma-3b-mix-448"
DEFAULT_BATCH_SIZE = 1   # keep batch=1 for safety on V100 16GB
MAX_NEW_TOKENS     = 64

SYSTEM_PROMPT = (
    "You are analyzing a Continuous Glucose Monitor (CGM) graph for a single day. "
    "The x-axis shows time (00:00-23:59) and the y-axis shows glucose level in mg/dL. "
    "The red dashed line marks 180 mg/dL (hyperglycemia threshold). "
    "The blue dashed line marks 70 mg/dL (hypoglycemia threshold). "
    "Answer concisely in the exact format requested."
)


def load_model(model_id: str):
    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # bfloat16: V100 (CC7.0) supports bf16; float32 Conv2d breaks cuDNN on cu118
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # disable FlashAttention2 (not supported on V100 CC7.0)
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded (bfloat16), device: {device}")
    return model, processor


def run_batch(model, processor, batch: list[dict]) -> list[str]:
    """Run inference on a batch; return list of answer strings."""
    answers = []
    for sample in batch:
        image_path = IMAGE_BASE / sample["image_file"]
        image = Image.open(image_path).convert("RGB")
        
        # PaliGemma requires <image> token at start of prompt
        prompt = f"<image>\n{SYSTEM_PROMPT}\nQuestion: {sample['question']}\nAnswer format: {sample['expected_answer_format']}\nAnswer:"

        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.0,
            )

        raw = processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )
        answers.append(raw.strip())

    return answers


def run_evaluation(model_id: str, batch_size: int):
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(EVAL_FILE) as f:
        samples = json.load(f)

    # Resume from checkpoint
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        done_ids = {r["sample_id"] for r in results}
        print(f"Resuming – {len(done_ids)}/{len(samples)} already done.")
    else:
        results  = []
        done_ids = set()

    # Skip samples with missing image_file
    skipped = [s for s in samples if not s.get("image_file")]
    if skipped:
        print(f"Skipping {len(skipped)} samples with no image_file: "
              f"{[s['sample_id'] for s in skipped]}")

    remaining = [s for s in samples
                 if s["sample_id"] not in done_ids and s.get("image_file")]
    if not remaining:
        print("All samples already evaluated.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, processor = load_model(model_id)

    total  = len(samples)
    errors = 0

    for batch_start in range(0, len(remaining), batch_size):
        batch     = remaining[batch_start: batch_start + batch_size]
        first_sid = batch[0]["sample_id"]
        last_sid  = batch[-1]["sample_id"]
        print(f"[{len(done_ids)+1:03d}-{len(done_ids)+len(batch):03d}/{total}] "
              f"{first_sid}..{last_sid} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            answers = run_batch(model, processor, batch)
            elapsed = round(time.time() - t0, 2)

            for sample, answer in zip(batch, answers):
                results.append({
                    "sample_id":    sample["sample_id"],
                    "graph_id":     sample["graph_id"],
                    "question_id":  sample["question_id"],
                    "task_type":    sample["task_type"],
                    "question":     sample["question"],
                    "model":        model_id,
                    "model_answer": answer,
                    "latency_s":    round(elapsed / len(batch), 2),
                    "timestamp":    datetime.now().isoformat(timespec="seconds"),
                })
                done_ids.add(sample["sample_id"])

            # Checkpoint after every batch
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"done [{elapsed}s, {elapsed/len(batch):.1f}s/sample]")
            for s, a in zip(batch, answers):
                print(f"    {s['sample_id']} ({s['task_type']}): \"{a[:80]}\"")

        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
            errors += len(batch)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Done. {len(results)} evaluated, {errors} errors.")
    print(f"Results: {RESULTS_FILE}")
    if results:
        avg_lat = sum(r["latency_s"] for r in results) / len(results)
        yn  = sum(1 for r in results if r["task_type"] == "yes_no")
        tmp = sum(1 for r in results if r["task_type"] == "temporal")
        print(f"yes_no={yn}  temporal={tmp}  avg latency={avg_lat:.2f}s/sample")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PaliGemma CGM-VQA Evaluation (HuggingFace + GPU)")
    parser.add_argument("--model",      default=DEFAULT_MODEL,
                        help="HuggingFace model ID")
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int,
                        help="Batch size for inference")
    args = parser.parse_args()

    run_evaluation(model_id=args.model, batch_size=args.batch_size)
