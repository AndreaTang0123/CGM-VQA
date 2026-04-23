import json
import re
import os

target_file = "results/qwen2vl7b_hf_results.json"
backup_file = "results/qwen2vl7b_hf_results.json.bak"

with open(target_file, 'r') as f:
    data = json.load(f)

# Create backup
with open(backup_file, 'w') as f:
    json.dump(data, f, indent=2)

count = 0
for item in data:
    orig = item.get("model_answer", "")
    # Replace HH.MM with HH:MM
    # Regex looks for 1-2 digits followed by a dot followed by exactly 2 digits
    # We ensure it's not a standard decimal if needed, but in this context it's likely time
    fixed = re.sub(r'(\d{1,2})\.(\d{2})', r'\1:\2', str(orig))
    
    # Also handle multiple dots or other common variations if they appear
    if fixed != orig:
        item["model_answer"] = fixed
        count += 1

with open(target_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Fixed {count} entries in {target_file}")
