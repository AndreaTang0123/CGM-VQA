import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def plot_accuracy():
    with open("statistics/yes_no_accuracy.json") as f:
        stats = json.load(f)
    
    models = sorted(stats.keys(), key=lambda x: stats[x]['accuracy'], reverse=True)
    accuracies = [stats[m]['accuracy'] for m in models]
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(models, accuracies, color='skyblue')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score (Accuracy)', fontsize=12)
    plt.title('Yes/No Accuracy by Model', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('statistics/yes_no_accuracy_bar_chart.png', dpi=300)
    plt.close()
    print("Generated statistics/yes_no_accuracy_bar_chart.png")

def plot_temporal_iou():
    with open("statistics/all_temporal_iou.json") as f:
        stats = json.load(f)
    
    models = sorted(stats.keys(), key=lambda x: stats[x]['mean_iou'], reverse=True)
    ious = [stats[m]['mean_iou'] for m in models]
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(models, ious, color='lightcoral')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Score (Mean IoU)', fontsize=12)
    plt.title('Temporal IoU by Model', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(ious) * 1.2 if ious else 0.1)
    
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('statistics/temporal_iou_bar_chart.png', dpi=300)
    plt.close()
    print("Generated statistics/temporal_iou_bar_chart.png")

def plot_latency_distribution():
    results_dir = "results"
    files = sorted(glob.glob(os.path.join(results_dir, "*_results.json")))
    
    all_latencies = []
    model_names = []
    
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        
        latencies = [item['latency_s'] for item in data if 'latency_s' in item]
        if latencies:
            model_name = os.path.basename(file).replace("_results.json", "")
            all_latencies.append(latencies)
            model_names.append(model_name)
    
    if not all_latencies:
        print("No latency data found.")
        return

    # Sort by median latency
    medians = [np.median(l) for l in all_latencies]
    sorted_indices = np.argsort(medians)
    
    all_latencies = [all_latencies[i] for i in sorted_indices]
    model_names = [model_names[i] for i in sorted_indices]

    plt.figure(figsize=(12, 8))
    
    # Create boxplot
    box = plt.boxplot(all_latencies, labels=model_names, vert=False, patch_artist=True, 
                      showfliers=False,
                      medianprops=dict(color="black", linewidth=2))
    
    # Color the boxes and add median text
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    median_texts = []
    for i, (patch, color) in enumerate(zip(box['boxes'], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
        # Add median label on box
        median_val = medians[sorted_indices[i]]
        plt.text(median_val, i + 1, f"{median_val:.2f}s", 
                 va='center', ha='left', fontsize=10, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        median_texts.append(f"{model_names[i]}: {median_val:.2f}s")
    
    # Add summary box in bottom right
    summary_text = "Medians:\n" + "\n".join(median_texts)
    plt.text(0.95, 0.05, summary_text, transform=plt.gca().transAxes,
             verticalalignment='bottom', horizontalalignment='right',
             fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.xlabel('Latency (Seconds per sample)', fontsize=12)
    plt.title('Latency Distribution by Model (Linear Scale)', fontsize=15)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('statistics/latency_distribution_boxplot.png', dpi=300)
    plt.close()
    print("Generated statistics/latency_distribution_boxplot.png")

def plot_mean_latency():
    results_dir = "results"
    files = sorted(glob.glob(os.path.join(results_dir, "*_results.json")))
    
    means = []
    model_names = []
    
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        
        latencies = [item['latency_s'] for item in data if 'latency_s' in item]
        if latencies:
            model_name = os.path.basename(file).replace("_results.json", "")
            means.append(np.mean(latencies))
            model_names.append(model_name)
    
    if not means:
        print("No latency data found.")
        return

    # Sort by mean latency
    sorted_indices = np.argsort(means)
    means = [means[i] for i in sorted_indices]
    model_names = [model_names[i] for i in sorted_indices]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(model_names, means, color='lightgreen')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Mean Latency (Seconds per sample)', fontsize=12)
    plt.title('Mean Latency by Model', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(means) * 1.15 if means else 5)
    
    # Add values on top
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('statistics/mean_latency_bar_chart.png', dpi=300)
    plt.close()
    print("Generated statistics/mean_latency_bar_chart.png")

def plot_accuracy_vs_latency():
    # Load Accuracy
    with open("statistics/yes_no_accuracy.json") as f:
        acc_stats = json.load(f)
    
    # Collect data
    results_dir = "results"
    models = sorted(acc_stats.keys())
    latencies = []
    accuracies = []
    
    valid_models = []
    for m in models:
        result_file = os.path.join(results_dir, f"{m}_results.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            l = [item['latency_s'] for item in data if 'latency_s' in item]
            if l:
                latencies.append(np.mean(l))
                accuracies.append(acc_stats[m]['accuracy'])
                valid_models.append(m)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(latencies, accuracies, color='blue', s=100, alpha=0.7)
    
    # Label points
    for i, model in enumerate(valid_models):
        plt.annotate(model, (latencies[i], accuracies[i]), xytext=(5, 5), 
                     textcoords='offset points', fontsize=10)
    
    plt.xlabel('Mean Latency (Seconds per sample)', fontsize=12)
    plt.ylabel('Yes/No Accuracy', fontsize=12)
    plt.title('Accuracy vs. Latency Trade-off', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('statistics/accuracy_vs_latency_scatter.png', dpi=300)
    plt.close()
    print("Generated statistics/accuracy_vs_latency_scatter.png")

def plot_iou_vs_latency():
    # Load IoU
    with open("statistics/all_temporal_iou.json") as f:
        iou_stats = json.load(f)
    
    # Collect data
    results_dir = "results"
    models = sorted(iou_stats.keys())
    latencies = []
    ious = []
    
    valid_models = []
    for m in models:
        result_file = os.path.join(results_dir, f"{m}_results.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            l = [item['latency_s'] for item in data if 'latency_s' in item]
            if l:
                latencies.append(np.mean(l))
                ious.append(iou_stats[m]['mean_iou'])
                valid_models.append(m)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(latencies, ious, color='green', s=100, alpha=0.7)
    
    # Label points
    for i, model in enumerate(valid_models):
        plt.annotate(model, (latencies[i], ious[i]), xytext=(5, 5), 
                     textcoords='offset points', fontsize=10)
    
    plt.xlabel('Mean Latency (Seconds per sample)', fontsize=12)
    plt.ylabel('Temporal Mean IoU', fontsize=12)
    plt.title('Temporal IoU vs. Latency Trade-off', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('statistics/iou_vs_latency_scatter.png', dpi=300)
    plt.close()
    print("Generated statistics/iou_vs_latency_scatter.png")

if __name__ == "__main__":
    plot_accuracy()
    plot_temporal_iou()
    plot_latency_distribution()
    plot_mean_latency()
    plot_accuracy_vs_latency()
    plot_iou_vs_latency()
