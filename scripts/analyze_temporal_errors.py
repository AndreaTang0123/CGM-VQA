import json
import re
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def classify_answer(answer):
    answer = str(answer).strip().lower()
    
    # regex for HH:MM
    time_pts = re.findall(r'\d{1,2}:\d{2}', answer)
    # regex for just numbers (partial)
    standalone_nums = re.findall(r'\b\d{1,2}\b', answer)
    
    # 1. Invalid Format: No time points and no standalone numbers of interest
    if not time_pts and not standalone_nums:
        return "invalid_format"
    
    # 2. Wrong Structure: Correct time format HH:MM but not the right count
    if len(time_pts) == 1:
        return "wrong_structure" # Only one point
    if len(time_pts) > 2:
        return "wrong_structure" # Multiple intervals or too many points
    
    # 3. Partial Format
    # Case: "12" or "12-13" (missing minutes)
    if not time_pts and standalone_nums:
        return "partial_format"
    
    # Case: "12:00 to 13:00" (has 2 points but uses "to" or other text)
    if len(time_pts) == 2:
        # Check if there is anything other than numbers, colons, and hyphens/dots/spaces
        # Standard format should be HH:MM-HH:MM or HH:MM - HH:MM
        if "to" in answer or "and" in answer or len(re.sub(r'[\d: \-\.]', '', answer)) > 0:
            # If it has 2 points but non-standard separators, it's partial format
            return "partial_format"
            
    # Default: if it doesn't clearly fit "correct" format but has points
    # (Actually, let's refine: if it has 2 pts and standard separator, it's correct)
    if len(time_pts) == 2:
        pattern = r'^\d{1,2}:\d{2}[ \-\.]+\d{1,2}:\d{2}$'
        if re.match(pattern, answer):
            return "correct_format"
        else:
            return "partial_format"

    return "parsing_failure"

def main():
    with open("annotation/annotations.json") as f:
        annotations = json.load(f)
    
    temporal_ids = [q["sample_id"] for q in annotations if q.get("task_type") == "temporal"]
    
    results_dir = "results"
    model_errors = {}
    
    categories = ["invalid_format", "partial_format", "wrong_structure", "parsing_failure"]
    
    for result_file in sorted(glob.glob(os.path.join(results_dir, "*_results.json"))):
        model_name = os.path.basename(result_file).replace("_results.json", "")
        with open(result_file) as f:
            results = json.load(f)
        
        errors = {cat: 0 for cat in categories}
        total_temporal = 0
        
        for r in results:
            if r["sample_id"] in temporal_ids:
                total_temporal += 1
                cls = classify_answer(r["model_answer"])
                if cls in errors:
                    errors[cls] += 1
        
        model_errors[model_name] = errors

    # Visualization
    models = list(model_errors.keys())
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, cat in enumerate(categories):
        vals = [model_errors[m][cat] for m in models]
        ax.bar(x + i*width - 1.5*width, vals, width, label=cat.replace("_", " ").title())
    
    ax.set_ylabel('Number of Errors')
    ax.set_title('Error Classification for Temporal Questions by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = "statistics/temporal_error_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"Results saved to {save_path}")
    
    # Save statistics to JSON for reference
    with open("statistics/temporal_error_stats.json", "w") as f:
        json.dump(model_errors, f, indent=4)

if __name__ == "__main__":
    main()
