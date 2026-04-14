"""
Batch evaluation of CGM-VQA using MatCha (google/pix2struct-chartqa-base) via HuggingFace + PyTorch (GPU).

Requirements:
    pip install transformers accelerate pillow torch protobuf sentencepiece

Results saved to: results/chartqa_hf_results.json

Usage:
    python3 run_eval_chartqa_hf.py
    python3 run_eval_chartqa_hf.py --batch-size 4 --model google/pix2struct-chartqa-base
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
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
IMAGE_BASE   = BASE_DIR / "graphs_cropped"
EVAL_FILE    = BASE_DIR / "metadata" / "eval_questions.json"
RESULTS_DIR  = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "chartqa_hf_results.json"

DEFAULT_MODEL      = "google/pix2struct-chartqa-base"
DEFAULT_BATCH_SIZE = 4   # MatCha is ~282M parameters, batching works well
MAX_NEW_TOKENS     = 64

SYSTEM_PROMPT = (
    "You are analyzing a Continuous Glucose Monitor (CGM) graph for a single day. "
    "The x-axis shows time (00:00-23:59) and the y-axis shows glucose level in mg/dL. "
    "The red dashed line marks 180 mg/dL (hyperglycemia threshold). "
    "The blue dashed line marks 70 mg/dL (hypoglycemia threshold). "
    "Answer concisely in the exact format requested."
)


def build_prompt(question: str, expected_format: str) -> str:
    """MatCha format: Just passing context and question as the text query."""
    return question


def load_model(model_id: str):
    print(f"Loading model: {model_id}")

    # Use float16 for acceleration on V100 GPU
    dtype = torch.float16

    processor = Pix2StructProcessor.from_pretrained(model_id)
    # MatCha is small enough to fit easily in a single 16GB GPU memory
    model = Pix2StructForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cuda:0"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded ({dtype}), device: {device}")
    return model, processor


def run_batch(model, processor, batch: list[dict]) -> list[str]:
    """Run inference on a batch; return list of answer strings."""
    
    texts = [build_prompt(s["question"], s["expected_answer_format"]) for s in batch]
    images = [Image.open(IMAGE_BASE / s["image_file"]).convert("RGB") for s in batch]

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt"
    )
    
    # Cast float32 inputs (e.g. image patches) to the model's dtype
    inputs = {
        k: v.to(dtype=model.dtype, device=model.device) if v.dtype in [torch.float32, torch.float64] else v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            # No repetition penalty initially needed for MatCha encoder-decoder architecture usually
        )

    # Encode-Decoder outputs: Decode directly (no input stripping needed)
    answers = processor.batch_decode(output_ids, skip_special_tokens=True)
    return [a.strip() for a in answers]


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
            print(f"ERROR: {e}")
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
        description="MatCha CGM-VQA Evaluation (HuggingFace + GPU)")
    parser.add_argument("--model",      default=DEFAULT_MODEL,
                        help="HuggingFace model ID")
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int,
                        help="Batch size for inference")
    args = parser.parse_args()

    run_evaluation(model_id=args.model, batch_size=args.batch_size)
