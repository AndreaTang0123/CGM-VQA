import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Data
models = [
    'InternVL2-4B', 'Qwen2-VL-7B', 'InternVL2-8B', 
    'ChartQA-HF', 'LLaVA-7B', 'Matcha-HF', 
    'LLaVA-Gemma-2B', 'Gemma-2B-4E'
]
accuracy = [0.88, 0.86, 0.80, 0.64, 0.56, 0.32, 0.28, 0.28]
iou = [0.0862, 0.0548, 0.0345, 0.0, 0.0508, 0.0, 0.0827, 0.0209]
latency = [1.96, 1.44, 6.08, 1.39, 1.72, 1.42, 2.33, 0.84]

# Set plot style for a clean look
plt.style.use('seaborn-v0_8-muted')
fig, ax1 = plt.subplots(figsize=(12, 7))

x = np.arange(len(models))
width = 0.25

# Plot Accuracy and IoU on the left axis
bars1 = ax1.bar(x - width/2, accuracy, width, label='Yes/No Accuracy', alpha=0.8)
bars2 = ax1.bar(x + width/2, iou, width, label='Mean IoU', alpha=0.8)

ax1.set_ylabel('Scores (Accuracy & IoU)')
ax1.set_ylim(0, 1.0)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')

# Create a second y-axis for Latency
ax2 = ax1.twinx()
line = ax2.plot(x, latency, color='red', marker='o', linewidth=2, label='Latency (s)')
ax2.set_ylabel('Latency (Seconds)')
ax2.set_ylim(0, max(latency) + 1)

# Add titles and legends
plt.title('CGM-VQA Model Performance Benchmark', fontsize=14, pad=20)
fig.tight_layout()

# Merge legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

# Add grid for better readability
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(bars1)
# For bars2, only label if non-zero to keep it clean
for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax1.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

save_path = 'statistics/model_performance_summary.png'
plt.savefig(save_path, dpi=300)
print(f"Plot saved to {save_path}")
