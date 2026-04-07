"""
Batch evaluation of CGM-VQA using LLaVA-1.5-7B via HuggingFace + PyTorch (GPU).

Requirements:
    pip install transformers accelerate pillow torch

Results saved to: results/llava7b_hf_results.json

Usage:
    python3 run_eval_llava7b_hf.py
    python3 run_eval_llava7b_hf.py --batch-size 4 --model llava-hf/llava-1.5-7b-hf
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, LlavaProcessor

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
IMAGE_BASE   = BASE_DIR / "graphs_cropped"
EVAL_FILE    = BASE_DIR / "metadata" / "eval_questions.json"
RESULTS_DIR  = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "llava7b_hf_results.json"

DEFAULT_MODEL      = "llava-hf/llava-1.5-7b-hf"
DEFAULT_BATCH_SIZE = 4
MAX_NEW_TOKENS     = 64

SYSTEM_PROMPT = (
    "You are analyzing a Continuous Glucose Monitor (CGM) graph for a single day. "
    "The x-axis shows time (00:00-23:59) and the y-axis shows glucose level in mg/dL. "
    "The red dashed line marks 180 mg/dL (hyperglycemia threshold). "
    "The blue dashed line marks 70 mg/dL (hypoglycemia threshold). "
    "Answer concisely in the exact format requested."
)


def build_prompt(question: str, expected_format: str) -> str:
    """LLaVA-1.5 conversation format: USER: <image>\n{text} ASSISTANT:"""
    text = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {question}\n"
        f"Answer format: {expected_format}\n"
        f"Answer:"
    )
    return f"USER: <image>\n{text} ASSISTANT:"


def load_model(model_id: str, device: torch.device):
    print(f"Loading model: {model_id}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",          # auto-shard across available GPUs
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"Model loaded on {device} ({dtype})")
    return model, processor


def run_batch(model, processor, batch: list[dict], device: torch.device) -> list[str]:
    """Run inference on a batch; return list of answer strings."""
    prompts = [build_prompt(s["question"], s["expected_answer_format"]) for s in batch]
    images  = [Image.open(IMAGE_BASE / s["image_file"]).convert("RGB") for s in batch]

    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    # Decode only the newly generated tokens (strip the prompt)
    input_len  = inputs["input_ids"].shape[1]
    new_tokens = output_ids[:, input_len:]
    answers    = processor.batch_decode(new_tokens, skip_special_tokens=True)
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

    remaining = [s for s in samples if s["sample_id"] not in done_ids]
    if not remaining:
        print("All samples already evaluated.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, processor = load_model(model_id, device)

    total  = len(samples)
    errors = 0

    # Process in batches
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start: batch_start + batch_size]
        first_sid = batch[0]["sample_id"]
        last_sid  = batch[-1]["sample_id"]
        print(f"[{len(done_ids)+1:03d}-{len(done_ids)+len(batch):03d}/{total}] "
              f"{first_sid}..{last_sid} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            answers = run_batch(model, processor, batch, device)
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
    parser = argparse.ArgumentParser(description="LLaVA-7B CGM-VQA Evaluation (HuggingFace + GPU)")
    parser.add_argument("--model",      default=DEFAULT_MODEL,      help="HuggingFace model ID")
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int, help="Batch size for inference")
    args = parser.parse_args()

    run_evaluation(model_id=args.model, batch_size=args.batch_size)
