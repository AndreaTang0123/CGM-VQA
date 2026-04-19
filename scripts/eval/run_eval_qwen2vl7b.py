"""
Batch evaluation of CGM-VQA using Hugging Face BLIP-2.
Results are saved to: results/hf_blip2_qwen2vl7b_results.json
"""

import json
import time
from pathlib import Path
from datetime import datetime

from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent.parent
IMAGE_BASE    = BASE_DIR / "graphs_cropped"
EVAL_FILE     = BASE_DIR / "metadata" / "eval_questions.json"
RESULTS_DIR   = BASE_DIR / "results"
RESULTS_FILE  = RESULTS_DIR / "hf_blip2_qwen2vl7b_results.json"
MODEL_NAME    = "Salesforce/blip2-flan-t5-base"
SLEEP_BETWEEN = 0.5   # seconds between requests

SYSTEM_PROMPT = (
    "You are analyzing a Continuous Glucose Monitor (CGM) graph for a single day. "
    "The x-axis shows time (00:00–23:59) and the y-axis shows glucose level in mg/dL. "
    "The red dashed line marks 180 mg/dL (hyperglycemia threshold). "
    "The blue dashed line marks 70 mg/dL (hypoglycemia threshold). "
    "Answer concisely and exactly in the requested format."
)


def build_prompt(question: str, expected_format: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {question}\n\n"
        f"Answer format: {expected_format}\n"
        f"Answer:"
    )


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {MODEL_NAME} on {device}...")
    processor = Blip2Processor.from_pretrained(MODEL_NAME)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    return processor, model, device


def query_model(processor, model, device, prompt: str, image_path: Path) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def run_evaluation():
    with open(EVAL_FILE) as f:
        samples = json.load(f)

    RESULTS_DIR.mkdir(exist_ok=True)

    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        done_ids = {r["sample_id"] for r in results}
        print(f"Resuming – {len(done_ids)} samples already done.")
    else:
        results = []
        done_ids = set()

    processor, model, device = load_model()

    total = len(samples)
    errors = 0

    for sample in samples:
        sid = sample["sample_id"]
        if sid in done_ids:
            continue

        image_path = IMAGE_BASE / sample["image_file"]
        if not image_path.exists():
            print(f"[WARN] Image not found: {image_path}")
            errors += 1
            continue

        prompt = build_prompt(sample["question"], sample["expected_answer_format"])

        try:
            print(f"[{len(done_ids)+1:03d}/{total}] {sid} ({sample['task_type']}) ...", flush=True)
            t0 = time.time()
            model_answer = query_model(processor, model, device, prompt, image_path)
            elapsed = round(time.time() - t0, 2)

            result = {
                "sample_id":    sid,
                "graph_id":     sample["graph_id"],
                "question_id":  sample["question_id"],
                "task_type":    sample["task_type"],
                "question":     sample["question"],
                "model":        MODEL_NAME,
                "model_answer": model_answer,
                "latency_s":    elapsed,
                "timestamp":    datetime.now().isoformat(timespec="seconds"),
            }
            results.append(result)
            done_ids.add(sid)

            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"→ \"{model_answer[:80]}\"  [{elapsed}s]")

        except Exception as e:
            print(f"[ERROR] {sid}: {e}")
            errors += 1

        time.sleep(SLEEP_BETWEEN)

    print(f"\n{'='*60}")
    print(f"Done. {len(results)} samples evaluated, {errors} errors.")
    print(f"Results saved to: {RESULTS_FILE}")

    yn_samples = [r for r in results if r["task_type"] == "yes_no"]
    tmp_samples = [r for r in results if r["task_type"] == "temporal"]
    avg_lat = sum(r["latency_s"] for r in results) / len(results) if results else 0
    print(f"yes_no: {len(yn_samples)}  temporal: {len(tmp_samples)}")
    print(f"Average latency: {avg_lat:.1f}s per sample")


if __name__ == "__main__":
    run_evaluation()
