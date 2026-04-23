import json

def load_results(filename):
    with open(filename) as f:
        return {r["sample_id"]: r["model_answer"] for r in json.load(f)}

def load_meta():
    with open("annotation/annotations.json") as f:
        return {q["sample_id"]: q for q in json.load(f)}

meta = load_meta()
res4b = load_results("results/internvl2_4b_hf_results.json")
res8b = load_results("results/internvl2_8b_hf_results.json")

print("Samples where 4B succeeded but 8B failed (Yes/No):")
print("-" * 50)
count = 0
for sid, q in meta.items():
    if q.get("task_type") == "yes_no":
        gt = q["answer"].strip().lower()
        ans4b = res4b.get(sid, "").strip().lower()
        ans8b = res8b.get(sid, "").strip().lower()
        
        # Simple parse for comparison
        p4b = "yes" if "yes" in ans4b and "no" not in ans4b else ("no" if "no" in ans4b and "yes" not in ans4b else ans4b)
        p8b = "yes" if "yes" in ans8b and "no" not in ans8b else ("no" if "no" in ans8b and "yes" not in ans8b else ans8b)
        
        if p4b == gt and p8b != gt:
            print(f"Sample: {sid} | Question: {q['question']}")
            print(f"GT: {gt} | 4B: {ans4b} | 8B: {ans8b}")
            print("-" * 30)
            count += 1
            if count >= 3: break

print("\nSamples where 4B had higher IoU than 8B (Temporal):")
print("-" * 50)
# We can use the processed IoU from statistics
with open("statistics/all_temporal_iou.json") as f:
    iou_stats = json.load(f)

iou4b = iou_stats["internvl2_4b_hf"]["per_sample_results"]
iou8b = iou_stats["internvl2_8b_hf"]["per_sample_results"]

count = 0
for sid in iou4b:
    if sid in iou8b:
        v4b = iou4b[sid]["iou"]
        v8b = iou8b[sid]["iou"]
        if v4b > v8b + 0.2:
            print(f"Sample: {sid} | Question: {iou4b[sid]['question']}")
            print(f"GT: {iou4b[sid]['gt_interval']} | 4B: {iou4b[sid]['pred_raw']} (IoU:{v4b})")
            print(f"8B: {iou8b[sid]['pred_raw']} (IoU:{v8b})")
            print("-" * 30)
            count += 1
            if count >= 3: break
