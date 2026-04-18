import json
import glob
import os

with open("annotation/annotations.json") as f:
    annotations = json.load(f)

q_dict = {q["sample_id"]: q for q in annotations if q.get("task_type") == "yes_no"}

stats = {}

for result_file in sorted(glob.glob("results/*_hf_results.json")):
    with open(result_file) as f:
        results = json.load(f)
    
    r_dict = {r["sample_id"]: r["model_answer"] for r in results}
    
    correct = 0
    total = 0
    
    for sid, q in q_dict.items():
        if sid in r_dict:
            total += 1
            gt = q["answer"].strip().lower()
            pred = r_dict[sid].strip().lower()
            if "yes" in pred and "no" not in pred:
                parsed_pred = "yes"
            elif "no" in pred and "yes" not in pred:
                parsed_pred = "no"
            else:
                parsed_pred = pred
            
            if gt == parsed_pred:
                correct += 1
    
    model_name = result_file.split("/")[-1].replace("_results.json", "")
    accuracy = correct / total if total > 0 else 0
    stats[model_name] = {
        "correct": correct,
        "total": total,
        "accuracy": round(accuracy, 4)
    }

# Ensure the output directory exists
os.makedirs("statistics", exist_ok=True)

# Save to JSON
with open("statistics/yes_no_accuracy.json", "w") as f:
    json.dump(stats, f, indent=4)

# Save to Markdown
with open("statistics/yes_no_accuracy.md", "w") as f:
    f.write("# Yes/No Questions Accuracy\n\n")
    f.write("| Model | Correct | Total | Accuracy |\n")
    f.write("| --- | --- | --- | --- |\n")
    for model, data in sorted(stats.items(), key=lambda item: item[1]['accuracy'], reverse=True):
        f.write(f"| {model} | {data['correct']} | {data['total']} | {data['accuracy']:.2%} |\n")

print("Statistics saved to statistics/yes_no_accuracy.json and statistics/yes_no_accuracy.md")
