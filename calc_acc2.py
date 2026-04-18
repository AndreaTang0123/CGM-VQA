import json
import glob

with open("annotation/annotations.json") as f:
    annotations = json.load(f)

q_dict = {q["sample_id"]: q for q in annotations if q.get("task_type") == "yes_no"}

for result_file in glob.glob("results/*_hf_results.json"):
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
    print(f"{model_name:>20} Yes/No Accuracy: {correct}/{total} = {correct/total if total else 0:.4f}")
