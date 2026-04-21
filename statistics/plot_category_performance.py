import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import csv
import re

# Set backend to Agg for headless environments
import matplotlib
matplotlib.use('Agg')

def time_to_minutes(t_str):
    try:
        h, m = map(int, t_str.strip().split(":"))
        return h * 60 + m
    except:
        return 0

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

def get_performance_data():
    # 1. Load Question Pool categories
    q_pool_path = "metadata/question_pool_2.csv"
    q_to_cat = {}
    with open(q_pool_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_to_cat[row['question_id']] = row['category']

    # 2. Load Annotations (Sample -> Question/GT)
    with open("annotation/annotations.json") as f:
        annotations = json.load(f)
    
    sample_to_meta = {}
    for entry in annotations:
        sid = entry['sample_id']
        qid = entry['question_id']
        qtype = entry.get('task_type', 'unknown')
        cat = q_to_cat.get(qid, 'unknown')
        
        sample_to_meta[sid] = {
            'qid': qid,
            'type': qtype,
            'category': cat,
            'gt_answer': entry.get('answer', ''),
            't_start': entry.get('t_start'),
            't_end': entry.get('t_end')
        }

    # 3. Process each model result
    results_dir = "results"
    model_files = sorted(glob.glob(os.path.join(results_dir, "*_hf_results.json")))
    
    all_categories = sorted(list(set(q_to_cat.values())))
    if 'unknown' in all_categories: all_categories.remove('unknown')
    
    model_performance = {} # model -> category -> [scores]

    for m_file in model_files:
        model_name = os.path.basename(m_file).replace("_results.json", "")
        with open(m_file) as f:
            results = json.load(f)
        
        scores_by_cat = {cat: [] for cat in all_categories}
        
        for r in results:
            sid = r['sample_id']
            if sid not in sample_to_meta: continue
            
            meta = sample_to_meta[sid]
            cat = meta['category']
            if cat == 'unknown': continue
            
            model_ans = r['model_answer']
            score = 0.0
            
            if meta['type'] == 'yes_no':
                gt = meta['gt_answer'].strip().lower()
                pred = str(model_ans).strip().lower()
                parsed_pred = ""
                if "yes" in pred and "no" not in pred: parsed_pred = "yes"
                elif "no" in pred and "yes" not in pred: parsed_pred = "no"
                else: parsed_pred = pred
                
                score = 1.0 if gt == parsed_pred else 0.0
                
            elif meta['type'] == 'temporal':
                p_start_str, p_end_str = parse_model_answer(model_ans)
                gt_start_str = meta['t_start']
                gt_end_str = meta['t_end']
                
                if p_start_str and p_end_str and gt_start_str and gt_end_str:
                    try:
                        p_start = time_to_minutes(p_start_str)
                        p_end = time_to_minutes(p_end_str)
                        if p_start > p_end: p_start, p_end = p_end, p_start
                        g_start = time_to_minutes(gt_start_str)
                        g_end = time_to_minutes(gt_end_str)
                        score = compute_iou(p_start, p_end, g_start, g_end)
                    except:
                        score = 0.0
                else:
                    score = 0.0
            
            scores_by_cat[cat].append(score)
        
        # Calculate mean per category
        model_performance[model_name] = {cat: (np.mean(scores_by_cat[cat]) if scores_by_cat[cat] else 0.0) for cat in all_categories}

    return model_performance, all_categories

def plot_heatmap(model_performance, categories):
    models = sorted(model_performance.keys())
    data = []
    for m in models:
        row = [model_performance[m][cat] for cat in categories]
        data.append(row)
    
    data = np.array(data)
    
    plt.figure(figsize=(12, 10))
    im = plt.imshow(data, cmap="YlGnBu")
    
    # Show all ticks and label them with the respective list entries
    plt.xticks(np.arange(len(categories)), labels=[c.replace('_', '\n') for c in categories])
    plt.yticks(np.arange(len(models)), labels=models)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(plt.gca().get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(models)):
        for j in range(len(categories)):
            text = plt.gca().text(j, i, f"{data[i, j]:.2f}",
                           ha="center", va="center", color="black", fontweight='bold')

    plt.title("Model Performance across Question Categories\n(Accuracy for Yes/No, IoU for Temporal)", fontsize=15, pad=20)
    plt.colorbar(im, label='Score')
    plt.tight_layout()
    plt.savefig('statistics/performance_heatmap.png', dpi=300)
    plt.close()
    print("Generated statistics/performance_heatmap.png")

def plot_grouped_bar(model_performance, categories):
    models = sorted(model_performance.keys())
    
    # Define custom colors for models
    MODEL_COLORS = {
        'qwen2vl7b_hf': 'gold',         # Qwen as Yellow/Gold
        'internvl2_4b_hf': '#1f77b4',   # Blue
        'internvl2_8b_hf': '#ff7f0e',   # Orange
        'llava7b_hf': '#2ca02c',        # Green
        'llava_gemma_2b_hf': '#d62728', # Red
        'chartqa_hf': '#9467bd',        # Purple
        'matcha_hf': '#8c564b',         # Brown
        'gemma4e2b_hf': '#e377c2',      # Pink
    }
    
    x = np.arange(len(categories))  # the label locations
    width = 0.8 / len(models)  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, m in enumerate(models):
        scores = [model_performance[m][cat] for cat in categories]
        offset = (i - len(models)/2) * width + width/2
        
        # Use custom color if defined, else use default cycle
        color = MODEL_COLORS.get(m)
        ax.bar(x + offset, scores, width, label=m, color=color)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score (Acc/IoU)')
    ax.set_title('Performance by Category and Model')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ') for c in categories])
    ax.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig('statistics/performance_grouped_bar.png', dpi=300)
    plt.close()
    print("Generated statistics/performance_grouped_bar.png")

if __name__ == "__main__":
    perf_data, cats = get_performance_data()
    plot_heatmap(perf_data, cats)
    plot_grouped_bar(perf_data, cats)
