import json
import base64
import os
import time
from pathlib import Path
from datetime import datetime

from openai import OpenAI

BASE_DIR = Path(__file__).parent.parent.parent
IMAGE_BASE = BASE_DIR / "graphs_cropped"
EVAL_FILE = BASE_DIR / "metadata" / "eval_questions.json"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "gpt4o_results.json"
JSONL_FILE = BASE_DIR / "scripts" / "eval" / "gpt4o_batch_requests.jsonl"
BATCH_ID_FILE = BASE_DIR / "scripts" / "eval" / ".gpt4o_batch_id"

SYSTEM_PROMPT = (
    "You are analyzing a Continuous Glucose Monitor (CGM) graph for a single day. "
    "The x-axis shows time (00:00-23:59) and the y-axis shows glucose level in mg/dL. "
    "The red dashed line marks 180 mg/dL (hyperglycemia threshold). "
    "The blue dashed line marks 70 mg/dL (hypoglycemia threshold). "
    "Answer concisely in the exact format requested."
)

def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def build_requests_jsonl():
    with open(EVAL_FILE, "r") as f:
        samples = json.load(f)
    
    requests = []
    for s in samples:
        if not s.get("image_file"):
            continue
            
        img_path = IMAGE_BASE / s["image_file"]
        base64_image = encode_image(img_path)
        
        prompt_text = (
            f"Question: {s['question']}\n"
            f"Answer format: {s['expected_answer_format']}\n"
            f"Answer:"
        )
        
        request = {
            "custom_id": str(s["sample_id"]),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.0
            }
        }
        requests.append(request)
        
    with open(JSONL_FILE, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
            
    return samples

def process_results(samples, results_lines):
    RESULTS_DIR.mkdir(exist_ok=True)
    
    sample_map = {str(s["sample_id"]): s for s in samples}
    results = []
    
    for line in results_lines:
        res = json.loads(line)
        custom_id = res["custom_id"]
        
        err = res["response"]["body"].get("error")
        if err:
            print(f"Error for {custom_id}: {err}")
            continue
            
        answer = res["response"]["body"]["choices"][0]["message"]["content"]
        s = sample_map.get(custom_id)
        if not s:
            continue
            
        results.append({
            "sample_id": s["sample_id"],
            "graph_id": s["graph_id"],
            "question_id": s["question_id"],
            "task_type": s["task_type"],
            "question": s["question"],
            "model": "gpt-4o",
            "model_answer": answer,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Done. Saved {len(results)} results to {RESULTS_FILE}")

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable (or put it in a .env file).")
        return

    client = OpenAI(api_key=api_key, max_retries=5)
    
    batch_id = None
    if BATCH_ID_FILE.exists():
        with open(BATCH_ID_FILE, "r") as f:
            batch_id = f.read().strip()
            
    samples = build_requests_jsonl()

    if not batch_id:
        print("Uploading requests file...")
        batch_input_file = client.files.create(
            file=open(JSONL_FILE, "rb"),
            purpose="batch"
        )
        print(f"File uploaded. ID: {batch_input_file.id}")
        
        print("Creating batch job...")
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "CGM-VQA Evaluation"}
        )
        batch_id = batch.id
        print(f"Batch created. ID: {batch_id}")
        
        with open(BATCH_ID_FILE, "w") as f:
            f.write(batch_id)
    else:
        print(f"Resuming batch job check for ID: {batch_id}")
        
    # Polling loop
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch status: {batch.status}")
        
        if batch.status in ['completed', 'failed', 'cancelled']:
            break
            
        time.sleep(30)
        
    if batch.status == 'completed':
        print("Batch completed! Downloading results...")
        output_file_id = batch.output_file_id
        file_response = client.files.content(output_file_id)
        
        lines = file_response.text.strip().split('\n')
        process_results(samples, lines)
        
        # Cleanup
        if BATCH_ID_FILE.exists():
            BATCH_ID_FILE.unlink()
    else:
        print(f"Batch ended with status: {batch.status}")
        if batch.errors:
            print("Errors:")
            for err in batch.errors:
                print(err)

if __name__ == "__main__":
    main()
