import json
import glob
import numpy as np

results_dir = "/nas/longleaf/home/txueying/CGM-VQA/results"
files = glob.glob(f"{results_dir}/*_results.json")

stats = {}

for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        
    latencies = [item['latency_s'] for item in data if 'latency_s' in item]
    
    if latencies:
        # Get model name from the file name or data
        model_name = file.split('/')[-1].replace("_results.json", "")
        
        mean_lat = np.mean(latencies)
        median_lat = np.median(latencies)
        p95_lat = np.percentile(latencies, 95)
        
        stats[model_name] = {
            "mean": mean_lat,
            "median": median_lat,
            "p95": p95_lat
        }

# Write results to a markdown file
md_content = "# Model Latency Statistics (Seconds per sample)\n\n"
md_content += "| Model | Mean Latency | Median Latency | P95 Latency |\n"
md_content += "| --- | --- | --- | --- |\n"

# Sort by Mean Latency (ascending)
sorted_stats = sorted(stats.items(), key=lambda x: x[1]['mean'])

for model, stat in sorted_stats:
    md_content += f"| {model} | {stat['mean']:.2f} s | {stat['median']:.2f} s | {stat['p95']:.2f} s |\n"

output_path = "/nas/longleaf/home/txueying/CGM-VQA/statistics/latency_stats.md"
with open(output_path, "w") as f:
    f.write(md_content)

print(f"Latency statistics generated at: {output_path}")
