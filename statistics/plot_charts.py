import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['internvl2_4b_hf', 'qwen2vl7b_hf', 'internvl2_8b_hf', 'chartqa_hf', 'llava7b_hf', 'matcha_hf', 'gemma4e2b_hf']
yes_no_acc = [0.88, 0.86, 0.80, 0.64, 0.56, 0.32, 0.28]
temporal_iou = [0.0862, 0.0548, 0.0345, 0.0000, 0.0508, 0.0000, 0.0209] # matching models order
# wait, temporal_iou order above from md:
# internvl2_4b: 0.0862
# qwen2vl7b: 0.0548
# llava7b: 0.0508
# internvl2_8b: 0.0345
# gemma4e2b: 0.0209
# chartqa: 0.0000
# matcha: 0.0000

# Mapping models to their temporal IoU
iou_dict = {
    'internvl2_4b_hf': 0.0862,
    'qwen2vl7b_hf': 0.0548,
    'llava7b_hf': 0.0508,
    'internvl2_8b_hf': 0.0345,
    'gemma4e2b_hf': 0.0209,
    'chartqa_hf': 0.0000,
    'matcha_hf': 0.0000
}

# Matching the order
temporal_iou_ordered = [iou_dict[m] for m in models]

# Plot Yes/No Accuracy
plt.figure(figsize=(10, 6))
bars = plt.bar(models, yes_no_acc, color='skyblue')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Score (Accuracy)', fontsize=12)
plt.title('Yes/No Accuracy by Model', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.0)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('/nas/longleaf/home/txueying/CGM-VQA/statistics/yes_no_accuracy_bar_chart.png')
plt.close()

# Plot Temporal IoU (Mean IoU)
models_iou_sorted = sorted(models, key=lambda x: iou_dict[x], reverse=True)
temporal_iou_sorted = [iou_dict[m] for m in models_iou_sorted]

plt.figure(figsize=(10, 6))
bars = plt.bar(models_iou_sorted, temporal_iou_sorted, color='lightcoral')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Score (Mean IoU)', fontsize=12)
plt.title('Temporal IoU by Model', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 0.1)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('/nas/longleaf/home/txueying/CGM-VQA/statistics/temporal_iou_bar_chart.png')
plt.close()

print("Charts successfully generated.")
