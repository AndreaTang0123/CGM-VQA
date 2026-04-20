import json
import re
import os
import glob
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

def time_to_minutes(t_str):
    h, m = map(int, t_str.strip().split(":"))
    return h * 60 + m

def parse_model_answer(answer_str):
    matches = re.findall(r'(\d{1,2}:\d{2})', str(answer_str))
    if len(matches) >= 2:
        return matches[0], matches[1]
    elif len(matches) == 1:
        return matches[0], matches[0]
    return None, None

def calculate_consistency_score(question_id, g_vals, t_vals):
    if len(g_vals) < 2:
        return 0
    
    g_start, g_end = g_vals[0], g_vals[-1]
    t_start, t_end = t_vals[0], t_vals[-1]
    dt_minutes = max((t_end - t_start), 1.0)
    overall_slope = (g_end - g_start) / dt_minutes
    diffs = np.diff(g_vals)
    
    # "rose most rapidly", "largest upward trend"
    if question_id in ["Q26", "Q28"]:
        return 1 if overall_slope > 0.5 else 0
        
    # "declined after an insulin event"
    elif question_id == "Q27":
        prop_neg = np.sum(diffs < 0) / len(diffs) if len(diffs) > 0 else 0
        return 1 if overall_slope < 0 and prop_neg >= 0.5 else 0
        
    # "gradually decreased"
    elif question_id == "Q29":
        prop_neg = np.sum(diffs < 0) / len(diffs) if len(diffs) > 0 else 0
        return 1 if overall_slope < 0 and prop_neg >= 0.5 else 0
        
    # "remained relatively stable"
    elif question_id == "Q30":
        std_dev = np.std(g_vals)
        return 1 if std_dev < 15.0 else 0
        
    # "continuously increased for at least 30 minutes"
    elif question_id == "Q40":
        prop_pos = np.sum(diffs > 0) / len(diffs) if len(diffs) > 0 else 0
        is_continuous = prop_pos >= 0.7
        is_long_enough = dt_minutes >= 30
        return 1 if (overall_slope > 0 and is_continuous and is_long_enough) else 0
        
    return None

def load_glucose_data(data_dir, filename):
    csv_name = filename.replace('.png', '.csv')
    path = os.path.join(data_dir, csv_name)
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Identify columns
        time_col, gluc_col = None, None
        for col in df.columns:
            if 'time' in col.lower() or 'datetime' in col.lower(): time_col = col
            if 'glucose' in col.lower() or 'value' in col.lower(): gluc_col = col
        if time_col and gluc_col:
            return df, time_col, gluc_col
            
    return None, None, None

def get_interval_data(df, time_col, gluc_col, start_min, end_min):
    if df is None: return [], []
    
    t_vals = []
    g_vals = []
    
    for _, row in df.iterrows():
        t_str = str(row[time_col])
        match = re.search(r'(\d{1,2}:\d{2})', t_str)
        if match:
            m = time_to_minutes(match.group(1))
            if start_min <= m <= end_min:
                t_vals.append(m)
                g_vals.append(float(row[gluc_col]))
                
    return g_vals, t_vals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Directory containing raw CGM CSV files (Readings_YYYY-MM-DD.csv)")
    args = parser.parse_args()

    # Load graphs mapping
    with open("metadata/cgm_vqa_base.json") as f:
        cgm_base = json.load(f)
        graph_metadata = {g["graph_id"]: g["file_name"] for g in cgm_base.get("graphs", [])}

    # Load annotations
    with open("annotation/annotations.json") as f:
        annotations = json.load(f)
        
    # We only care about specific questions based on ID
    target_qids = {"Q26", "Q27", "Q28", "Q29", "Q30", "Q40"}
    
    trend_samples = {}
    for q in annotations:
        if q.get("question_id") in target_qids:
            trend_samples[q["sample_id"]] = {
                "graph_id": q["graph_id"],
                "question_id": q["question_id"],
                "question": q["question"]
            }
                
    all_stats = {}
    
    # Parse results
    for result_file in sorted(glob.glob("results/*_results.json")):
        with open(result_file) as f:
            results = json.load(f)
            
        model_name = os.path.basename(result_file).replace("_results.json", "")
        pred_dict = {r["sample_id"]: r["model_answer"] for r in results}
        
        scores = []
        per_sample_stats = {}
        
        for sid, gt in trend_samples.items():
            if sid in pred_dict:
                pred_ans = pred_dict[sid]
                p_start_str, p_end_str = parse_model_answer(pred_ans)
                
                if p_start_str and p_end_str:
                    try:
                        p_start = time_to_minutes(p_start_str)
                        p_end = time_to_minutes(p_end_str)
                        if p_start > p_end: p_start, p_end = p_end, p_start
                        
                        g_file = graph_metadata.get(gt["graph_id"])
                        if not g_file: continue
                        
                        df, time_col, gluc_col = load_glucose_data(args.data_dir, g_file)
                        if df is None:
                            continue # Skip if no data loaded
                            
                        g_vals, t_vals = get_interval_data(df, time_col, gluc_col, p_start, p_end)
                        
                        score = calculate_consistency_score(gt["question_id"], g_vals, t_vals)
                        if score is not None:
                            scores.append(score)
                            per_sample_stats[sid] = {
                                "question_id": gt["question_id"],
                                "pred_interval": f"{p_start_str}-{p_end_str}",
                                "consistency_score": score
                            }
                    except ValueError:
                        pass
                        
        mean_score = float(np.mean(scores)) if scores else 0.0
        
        all_stats[model_name] = {
            "mean_consistency_score": round(mean_score, 4),
            "total_samples": len(scores),
            "per_sample_results": per_sample_stats
        }

    os.makedirs("statistics", exist_ok=True)
    
    with open("statistics/all_trend_consistency.json", "w") as f:
        json.dump(all_stats, f, indent=4)
        
    with open("statistics/trend_consistency.md", "w") as f:
        f.write("# Trend Consistency Scores\n\n")
        f.write("This table shows the 'Trend Consistency Score' for each model on temporal questions measuring graphical trends. Consistency is scored as 1 (satisfies trend) or 0.\n\n")
        f.write("| Model | Valid Trend Samples | Mean Trend Consistency |\n")
        f.write("| --- | --- | --- |\n")
        
        for model_name, data in sorted(all_stats.items(), key=lambda item: item[1]['mean_consistency_score'], reverse=True):
            f.write(f"| {model_name} | {data['total_samples']} | {data['mean_consistency_score']:.4f} |\n")
            
    print("Saved generic trend consistency statistics to statistics/all_trend_consistency.json and statistics/trend_consistency.md")

if __name__ == "__main__":
    main()
