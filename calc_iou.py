import json
import re
import os
import glob

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

def main():
    with open("annotation/annotations.json") as f:
        annotations = json.load(f)
        
    gt_dict = {q["sample_id"]: q for q in annotations if q.get("task_type") == "temporal"}
    
    all_stats = {}
    
    for result_file in sorted(glob.glob("results/*_results.json")):
        with open(result_file) as f:
            results = json.load(f)
            
        model_name = os.path.basename(result_file).replace("_results.json", "")
        pred_dict = {r["sample_id"]: r["model_answer"] for r in results}
        
        total = 0
        ious = []
        per_sample_iou = {}

        for sid, gt in gt_dict.items():
            if sid in pred_dict:
                total += 1
                pred_ans = pred_dict[sid]
                
                p_start_str, p_end_str = parse_model_answer(pred_ans)
                
                if not gt.get("t_start") or not gt.get("t_end"):
                    continue
                    
                gt_start_str = gt["t_start"]
                gt_end_str   = gt["t_end"]
                
                if p_start_str and p_end_str:
                    try:
                        p_start = time_to_minutes(p_start_str)
                        p_end   = time_to_minutes(p_end_str)
                        if p_start > p_end:
                            p_start, p_end = p_end, p_start
                            
                        g_start = time_to_minutes(gt_start_str)
                        g_end   = time_to_minutes(gt_end_str)
                        
                        iou = compute_iou(p_start, p_end, g_start, g_end)
                    except ValueError:
                        iou = 0.0
                else:
                    iou = 0.0
                    
                ious.append(iou)
                per_sample_iou[sid] = {
                    "question": gt["question"],
                    "gt_interval": f"{gt_start_str}-{gt_end_str}",
                    "pred_raw": pred_ans,
                    "iou": round(iou, 4)
                }
                
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        iou_acc = sum(1 for i in ious if i >= 0.5) / len(ious) if ious else 0.0
        
        all_stats[model_name] = {
            "mean_iou": round(mean_iou, 4),
            "iou_acc": round(iou_acc, 4),
            "total_samples": total,
            "per_sample_results": per_sample_iou
        }
    
    os.makedirs("statistics", exist_ok=True)
    
    with open("statistics/all_temporal_iou.json", "w") as f:
        json.dump(all_stats, f, indent=4)
        
    with open("statistics/temporal_iou.md", "w") as f:
        f.write("# Temporal Questions IoU Accuracy\n\n")
        f.write("| Model | Temporal Samples | Mean IoU | Accuracy (IoU ≥ 0.5) |\n")
        f.write("| --- | --- | --- | --- |\n")
        for model_name, data in sorted(all_stats.items(), key=lambda item: item[1]['mean_iou'], reverse=True):
            f.write(f"| {model_name} | {data['total_samples']} | {data['mean_iou']:.4f} | {data['iou_acc']:.2%} |\n")
            
    print("Saved aggregate statistics to statistics/all_temporal_iou.json and statistics/temporal_iou.md")

if __name__ == "__main__":
    main()
