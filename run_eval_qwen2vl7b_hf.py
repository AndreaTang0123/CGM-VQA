"""
Batch evaluation of CGM-VQA using Qwen2-VL-7B-Instruct via HuggingFace + PyTorch (GPU).

Requirements:
    pip install transformers accelerate pillow torch qwen-vl-utils bitsandbytes protobuf

Results saved to: results/qwen2vl7b_hf_results.json

Usage:
    python3 run_eval_qwen2vl7b_hf.py
    python3 run_eval_qwen2vl7b_hf.py --batch-size 1 --model Qwen/Qwen2-VL-7B-Instruct
"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Synchronous CUDA errors — critical on V100 CC7.0
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
# Limit OpenMP/MKL threads to avoid over-subscription on HPC nodes
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
IMAGE_BASE   = BASE_DIR / "graphs_cropped"
EVAL_FILE    = BASE_DIR / "metadata" / "eval_questions.json"
RESULTS_DIR  = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "qwen2vl7b_hf_results.json"

DEFAULT_MODEL      = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_BATCH_SIZE = 1   # V100 16GB fills up quickly with a 7B VL model
MAX_NEW_TOKENS     = 64

# Cap image resolution for V100 CC7.0 (limits token count from dynamic patching)
# V100 has a 1024-thread-per-block limit; large RoPE kernels exceed this.
# Reducing MAX_PIXELS keeps sequence length short → smaller kernel thread blocks.
# 84x28x28 = 65856 ≈ 256x256 image  (previously 256x28x28 caused kernel errors)
MIN_PIXELS = 4 * 28 * 28   #   3136
MAX_PIXELS = 84 * 28 * 28  #  65856  (~256×256) — V100-safe

SYSTEM_PROMPT = (
    "You are analyzing a Continuous Glucose Monitor (CGM) graph for a single day. "
    "The x-axis shows time (00:00-23:59) and the y-axis shows glucose level in mg/dL. "
    "The red dashed line marks 180 mg/dL (hyperglycemia threshold). "
    "The blue dashed line marks 70 mg/dL (hypoglycemia threshold). "
    "Answer concisely in the exact format requested."
)


def build_messages(question: str, expected_format: str, image_path: Path) -> list:
    """Qwen2-VL chat format with interleaved image."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {
                    "type": "text",
                    "text": (
                        f"{SYSTEM_PROMPT}\n\n"
                        f"Question: {question}\n"
                        f"Answer format: {expected_format}"
                    ),
                },
            ],
        }
    ]


def load_model(model_id: str):
    print(f"Loading model: {model_id}")

    # V100 (CC 7.0) does not support bfloat16; use float16
    dtype = torch.float16

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    # Using 2 GPUs horizontally sharded by accelerate via device_map="auto"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",             # lets accelerate assign layers across 2 GPUs
        attn_implementation="sdpa",    # SDPA is natively supported by PyTorch 2.x and much more stable on V100 than eager
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded (INT8 quantized), device: {device}")
    return model, processor


def run_batch(model, processor, batch: list[dict]) -> list[str]:
    """Run inference on a batch; return list of answer strings."""
    from qwen_vl_utils import process_vision_info

    all_messages = [
        build_messages(s["question"], s["expected_answer_format"],
                       IMAGE_BASE / s["image_file"])
        for s in batch
    ]

    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]
    image_inputs, video_inputs = process_vision_info(all_messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    ).to(model.device)

    with torch.no_grad():
        try:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.05,   # prevents ![](![]... loops on V100/float16
                no_repeat_ngram_size=3,    # blocks trigram repetition
            )
        except RuntimeError as e:
            if "too many resources" in str(e) or "CUDA error" in str(e):
                # Clear CUDA state and retry with cleared cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    repetition_penalty=1.05,
                    no_repeat_ngram_size=3,
                )
            else:
                raise

    # Decode only newly generated tokens
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
        description="Qwen2-VL-7B CGM-VQA Evaluation (HuggingFace + GPU)")
    parser.add_argument("--model",      default=DEFAULT_MODEL,
                        help="HuggingFace model ID")
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int,
                        help="Batch size for inference")
    args = parser.parse_args()

    run_evaluation(model_id=args.model, batch_size=args.batch_size)
