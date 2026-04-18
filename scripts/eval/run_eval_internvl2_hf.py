"""
Batch evaluation of CGM-VQA using InternVL2-4B via HuggingFace + PyTorch (GPU).

Requirements:
    pip install transformers accelerate torchvision pillow torch einops timm

Results saved to: results/internvl2_4b_hf_results.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
IMAGE_BASE   = BASE_DIR / "graphs_cropped"
EVAL_FILE    = BASE_DIR / "metadata" / "eval_questions.json"
RESULTS_DIR  = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "internvl2_4b_hf_results.json"

DEFAULT_MODEL      = "OpenGVLab/InternVL2-4B"
DEFAULT_BATCH_SIZE = 1  
MAX_NEW_TOKENS     = 64

SYSTEM_PROMPT = (
    "You are analyzing a Continuous Glucose Monitor (CGM) graph for a single day. "
    "The x-axis shows time (00:00-23:59) and the y-axis shows glucose level in mg/dL. "
    "The red dashed line marks 180 mg/dL (hyperglycemia threshold). "
    "The blue dashed line marks 70 mg/dL (hypoglycemia threshold). "
    "Answer concisely in the exact format requested."
)

# ── InternVL2 Specific Preprocessing ─────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ── Main ──────────────────────────────────────────────────────────────────────

def load_model_internvl(model_id: str, device: torch.device):
    print(f"Loading model: {model_id}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Disable flash attn since V100 does not support it (Compute Capability 7.0)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False, # explicit false for V100 compatibility
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    print(f"Model loaded on {device} ({dtype})")
    return model, tokenizer

def run_evaluation(model_id: str, batch_size: int):
    RESULTS_DIR.mkdir(exist_ok=True)

    model_name_safe = model_id.split("/")[-1].lower().replace("-", "_")
    results_file = RESULTS_DIR / f"{model_name_safe}_hf_results.json"

    with open(EVAL_FILE) as f:
        samples = json.load(f)

    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        done_ids = {r["sample_id"] for r in results}
        print(f"Resuming – {len(done_ids)}/{len(samples)} already done.")
    else:
        results  = []
        done_ids = set()

    skipped = [s for s in samples if not s.get("image_file")]
    if skipped:
        print(f"Skipping {len(skipped)} samples with no image_file")
    remaining = [s for s in samples if s["sample_id"] not in done_ids and s.get("image_file")]
    if not remaining:
        print("All samples already evaluated.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, tokenizer = load_model_internvl(model_id, device)

    total  = len(samples)
    errors = 0

    generation_config = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start: batch_start + batch_size]
        first_sid = batch[0]["sample_id"]
        last_sid  = batch[-1]["sample_id"]
        print(f"[{len(done_ids)+1:03d}-{len(done_ids)+len(batch):03d}/{total}] "
              f"{first_sid}..{last_sid} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            pixel_values_list = []
            num_patches_list = []
            questions = []
            
            for s in batch:
                # max_num=6 limits image chunks to conserve 16GB VRAM on V100
                px = load_image(IMAGE_BASE / s["image_file"], max_num=6).to(model.dtype).to(device)
                pixel_values_list.append(px)
                num_patches_list.append(px.shape[0])
                
                q_text = (
                    f"<image>\n"
                    f"{SYSTEM_PROMPT}\n\n"
                    f"Question: {s['question']}\n"
                    f"Answer format: {s['expected_answer_format']}\n"
                    f"Answer:"
                )
                questions.append(q_text)
                
            pixel_values = torch.cat(pixel_values_list, dim=0)

            if len(batch) == 1:
                answer = model.chat(tokenizer, pixel_values, questions[0], generation_config)
                answers = [answer.strip()]
            else:
                answers = model.batch_chat(
                    tokenizer, 
                    pixel_values,
                    num_patches_list=num_patches_list,
                    questions=questions,
                    generation_config=generation_config
                )
                answers = [a.strip() for a in answers]

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

            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"done [{elapsed}s]")

        except Exception as e:
            print(f"ERROR: {e}")
            errors += len(batch)

    print(f"\nDone. {len(remaining)} processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", default=DEFAULT_BATCH_SIZE, type=int)
    args = parser.parse_args()
    run_evaluation(model_id=args.model, batch_size=args.batch_size)
