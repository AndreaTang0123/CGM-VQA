import csv
import matplotlib.pyplot as plt
import os

# File paths
csv_path = '/nas/longleaf/home/txueying/CGM-VQA/metadata/question_pool_2.csv'
output_dir = '/nas/longleaf/home/txueying/CGM-VQA/statistics'
output_image = os.path.join(output_dir, 'question_category_pie_chart.png')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define mapping
mapping = {
    'trend': 'trend_understanding',
    'duration': 'duration_reasoning',
    'event': 'event_localization',
    'comparison': 'comparison',
    'other': 'event_localization'
}

category_counts = {}

# Read and classify
with open(csv_path, mode='r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        old_cat = row['category']
        new_cat = mapping.get(old_cat, old_cat)
        category_counts[new_cat] = category_counts.get(new_cat, 0) + 1

# Prepare data for plotting
labels = list(category_counts.keys())
sizes = list(category_counts.values())

# Set style
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']
plt.figure(figsize=(10, 7))

# Create pie chart
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors[:len(labels)], shadow=False)
plt.title('Question Pool Category Distribution', fontsize=16, pad=20)
plt.axis('equal') 

# Add a legend
plt.legend(labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Save the plot
plt.tight_layout()
plt.savefig(output_image, dpi=300)
plt.close()

print(f"Pie chart saved to {output_image}")
print("\nCategory Counts:")
for cat, count in category_counts.items():
    print(f"{cat}: {count}")

# Update the CSV file with new categories
rows = []
fieldnames = []
with open(csv_path, mode='r', newline='') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        row['category'] = mapping.get(row['category'], row['category'])
        rows.append(row)

with open(csv_path, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated {csv_path} with new category names.")
