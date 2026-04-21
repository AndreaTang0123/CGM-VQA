import json
import glob
import os
import re

def time_to_minutes(t_str):
    h, m = map(int, t_str.strip().split(":"))
    return h * 60 + m

def compute_iou(pred_start, pred_end, gt_start, gt_end):
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0, inter_end - inter_start)
    
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = max(0, union_end - union_start)
    
    if union == 0:
        return 1.0 if pred_start == gt_start else 0.0
    return intersection / union

def parse_model_answer(answer_str):
    matches = re.findall(r'(\d{1,2}:\d{2})', str(answer_str))
    if len(matches) >= 2:
        return matches[0], matches[1]
    elif len(matches) == 1:
        return matches[0], matches[0]
    return None, None

def calculate_stats(dir_path, annotations):
    q_dict_yn = {q["sample_id"]: q for q in annotations if q.get("task_type") == "yes_no"}
    q_dict_temp = {q["sample_id"]: q for q in annotations if q.get("task_type") == "temporal"}
    
    all_stats = {}
    
    for result_file in sorted(glob.glob(os.path.join(dir_path, "*_hf_results.json"))):
        with open(result_file) as f:
            results = json.load(f)
        
        model_name = os.path.basename(result_file).replace("_results.json", "")
        r_dict = {r["sample_id"]: r["model_answer"] for r in results}
        
        # Yes/No Accuracy
        yn_correct = 0
        yn_total = 0
        for sid, q in q_dict_yn.items():
            if sid in r_dict:
                yn_total += 1
                gt = q["answer"].strip().lower()
                pred = r_dict[sid].strip().lower()
                if "yes" in pred and "no" not in pred:
                    parsed_pred = "yes"
                elif "no" in pred and "yes" not in pred:
                    parsed_pred = "no"
                else:
                    parsed_pred = pred
                
                if gt == parsed_pred:
                    yn_correct += 1
        
        yn_acc = yn_correct / yn_total if yn_total > 0 else 0
        
        # Temporal IoU
        temp_ious = []
        temp_total = 0
        for sid, gt in q_dict_temp.items():
            if sid in r_dict:
                temp_total += 1
                pred_ans = r_dict[sid]
                p_start_str, p_end_str = parse_model_answer(pred_ans)
                
                if not gt.get("t_start") or not gt.get("t_end"):
                    continue
                
                gt_start_str = gt["t_start"]
                gt_end_str = gt["t_end"]
                
                if p_start_str and p_end_str:
                    try:
                        p_start = time_to_minutes(p_start_str)
                        p_end = time_to_minutes(p_end_str)
                        if p_start > p_end:
                            p_start, p_end = p_end, p_start
                        g_start = time_to_minutes(gt_start_str)
                        g_end = time_to_minutes(gt_end_str)
                        iou = compute_iou(p_start, p_end, g_start, g_end)
                    except ValueError:
                        iou = 0.0
                else:
                    iou = 0.0
                temp_ious.append(iou)
        
        mean_iou = sum(temp_ious) / len(temp_ious) if temp_ious else 0.0
        iou_acc = sum(1 for i in temp_ious if i >= 0.5) / len(temp_ious) if temp_ious else 0.0
        
        all_stats[model_name] = {
            "yn_accuracy": yn_acc,
            "mean_iou": mean_iou,
            "iou_accuracy": iou_acc
        }
    
    return all_stats

if __name__ == "__main__":
    with open("annotation/annotations.json") as f:
        annotations = json.load(f)
    
    print("Calculating stats for current results...")
    current_stats = calculate_stats("results", annotations)
    
    print("\nCalculating stats for backup results...")
    backup_stats = calculate_stats("results_backup_20260419000539", annotations)
    
    print("\nComparison Results:")
    models = sorted(set(list(current_stats.keys()) + list(backup_stats.keys())))
    
    print(f"{'Model':<20} | {'Metric':<15} | {'Current':<10} | {'Backup':<10} | {'Diff':<10}")
    print("-" * 75)
    
    for model in models:
        c = current_stats.get(model, {})
        b = backup_stats.get(model, {})
        
        for metric, key in [("Yes/No Acc", "yn_accuracy"), ("Mean IoU", "mean_iou"), ("IoU Acc", "iou_accuracy")]:
            val_c = c.get(key, 0)
            val_b = b.get(key, 0)
            diff = val_c - val_b
            print(f"{model:<20} | {metric:<15} | {val_c:<10.4f} | {val_b:<10.4f} | {diff:<+10.4f}")
        print("-" * 75)
