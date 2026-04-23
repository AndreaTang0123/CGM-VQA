import json
import re
import os

target_file = "results/internvl2_8b_hf_results.json"
backup_file = "results/internvl2_8b_hf_results.json.bak"

with open(target_file, 'r') as f:
    data = json.load(f)

# Load annotations to identify temporal tasks
with open("annotation/annotations.json") as f:
    annotations = json.load(f)
temporal_ids = {q["sample_id"] for q in annotations if q.get("task_type") == "temporal"}

# Backup
with open(backup_file, 'w') as f:
    json.dump(data, f, indent=2)

def extract_best_interval(text):
    text = str(text).strip()
    
    # 1. Look for "Answer: HH:MM-HH:MM"
    ans_match = re.search(r'Answer:\s*(\d{1,2}:\d{2})\s*[\-到to ]+\s*(\d{1,2}:\d{2})', text, re.IGNORECASE)
    if ans_match:
        return f"{ans_match.group(1)}-{ans_match.group(2)}"
    
    # 2. Find all HH:MM patterns
    time_pts = re.findall(r'(\d{1,2}:\d{2})', text)
    
    if len(time_pts) >= 2:
        # If there are 4 or more points, it might be [event_start, event_end, answer_start, answer_end]
        # In many CoT cases, the last interval is the final answer.
        # Let's take the last two points.
        return f"{time_pts[-2]}-{time_pts[-1]}"
    elif len(time_pts) == 1:
        return f"{time_pts[0]}-{time_pts[0]}"
    
    return text # Fallback

count = 0
for item in data:
    if item["sample_id"] in temporal_ids:
        orig = item.get("model_answer", "")
        # Only clean if it's not already a simple clean format
        if len(str(orig)) > 15 or " " in str(orig):
            fixed = extract_best_interval(orig)
            if fixed != orig:
                item["model_answer"] = fixed
                count += 1

with open(target_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Cleaned {count} temporal entries in {target_file}")
