"""
Batch evaluation of CGM-VQA using ollama qwen2.5vl:7b
Results are saved to: results/qwen2vl7b_results.json
"""

import json
import base64
import time
import urllib.request
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
IMAGE_BASE    = BASE_DIR / "graphs_cropped"
EVAL_FILE     = BASE_DIR / "metadata" / "eval_questions.json"
RESULTS_DIR   = BASE_DIR / "results"
RESULTS_FILE  = RESULTS_DIR / "qwen2vl7b_results.json"
OLLAMA_URL    = "http://localhost:11434/api/generate"
MODEL         = "qwen2.5vl:7b"
SLEEP_BETWEEN = 0.5   # seconds between requests

SYSTEM_PROMPT = (
    "You are analyzing a Continuous Glucose Monitor (CGM) graph for a single day. "
    "The x-axis shows time (00:00–23:59) and the y-axis shows glucose level in mg/dL. "
    "The red dashed line marks 180 mg/dL (hyperglycemia threshold). "
    "The blue dashed line marks 70 mg/dL (hypoglycemia threshold). "
    "Answer concisely and exactly in the requested format."
)


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(question: str, expected_format: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {question}\n\n"
        f"Answer format: {expected_format}\n"
        f"Answer:"
    )


def query_model(prompt: str, b64_image: str) -> dict:
    payload = json.dumps({
        "model":  MODEL,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_evaluation():
    with open(EVAL_FILE) as f:
        samples = json.load(f)

    RESULTS_DIR.mkdir(exist_ok=True)

    # Resume from checkpoint if partial results exist
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        done_ids = {r["sample_id"] for r in results}
        print(f"Resuming – {len(done_ids)}/{len(samples)} already done.")
    else:
        results = []
        done_ids = set()

    total  = len(samples)
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
            print(f"[{len(done_ids)+1:03d}/{total}] {sid} ({sample['task_type']}) ...", end=" ", flush=True)
            b64     = encode_image(image_path)
            t0      = time.time()
            raw     = query_model(prompt, b64)
            elapsed = round(time.time() - t0, 2)

            model_answer = raw.get("response", "").strip()

            result = {
                "sample_id":    sid,
                "graph_id":     sample["graph_id"],
                "question_id":  sample["question_id"],
                "task_type":    sample["task_type"],
                "question":     sample["question"],
                "model":        MODEL,
                "model_answer": model_answer,
                "latency_s":    elapsed,
                "timestamp":    datetime.now().isoformat(timespec="seconds"),
            }
            results.append(result)
            done_ids.add(sid)

            # Save checkpoint after every sample
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f'"{model_answer[:70]}"  [{elapsed}s]')

        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

        time.sleep(SLEEP_BETWEEN)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Done. {len(results)} evaluated, {errors} errors.")
    print(f"Results: {RESULTS_FILE}")
    if results:
        avg_lat = sum(r["latency_s"] for r in results) / len(results)
        yn  = sum(1 for r in results if r["task_type"] == "yes_no")
        tmp = sum(1 for r in results if r["task_type"] == "temporal")
        print(f"yes_no={yn}  temporal={tmp}  avg latency={avg_lat:.1f}s")


if __name__ == "__main__":
    run_evaluation()
