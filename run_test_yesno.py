import os
import json
import time
from pathlib import Path

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

BASE_DIR     = Path(__file__).parent
IMAGE_BASE   = BASE_DIR / "graphs_cropped"
EVAL_FILE    = BASE_DIR / "metadata" / "eval_questions.json"
RESULTS_DIR  = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "chartqa_hf_yesno_test.json"

DEFAULT_MODEL      = "google/pix2struct-chartqa-base"

def load_model(model_id: str):
    print(f"Loading model: {model_id}")
    dtype = torch.float32
    processor = Pix2StructProcessor.from_pretrained(model_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Pix2StructForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device
    )
    model.eval()
    return model, processor

def build_prompt(question: str) -> str:
    """MatCha format: strictly raw query!"""
    return question

def run_test():
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(EVAL_FILE) as f:
        samples = json.load(f)

    # Filter for yes_no questions ONLY and take the first 10
    yes_no_samples = [s for s in samples if s["task_type"] == "yes_no" and s.get("image_file")]
    test_batch = yes_no_samples[:10]

    print(f"Testing on {len(test_batch)} yes_no samples...")

    model, processor = load_model(DEFAULT_MODEL)
    
    results = []
    texts = [build_prompt(s["question"]) for s in test_batch]
    images = [Image.open(IMAGE_BASE / s["image_file"]).convert("RGB") for s in test_batch]

    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt"
    )
    
    inputs = {
        k: v.to(dtype=model.dtype, device=model.device) if v.dtype in [torch.float32, torch.float64] else v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )

    answers = processor.batch_decode(output_ids, skip_special_tokens=True)
    answers = [a.strip() for a in answers]

    for sample, ans in zip(test_batch, answers):
        results.append({
            "sample_id": sample["sample_id"],
            "question": sample["question"],
            "model_answer": ans
        })
        print(f"[{sample['sample_id']}] Q: {sample['question']}")
        print(f"       -> A: {ans}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved results to {RESULTS_FILE}")

if __name__ == "__main__":
    run_test()
