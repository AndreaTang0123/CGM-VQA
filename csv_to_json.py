import csv
import json

graph_csv = "metadata/graph_metadata.csv"
question_csv = "metadata/question_pool.csv"
output_json = "metadata/cgm_vqa_base.json"

graphs = []
questions = []

with open(graph_csv, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        graphs.append({
            "graph_id": row["graph_id"],
            "file_name": row["file_name"]
        })

with open(question_csv, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        questions.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "type": row["type"],
            "category": row["category"]
        })

dataset = {
    "dataset": "CGM-VQA",
    "graphs": graphs,
    "question_pool": questions
}

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Saved {output_json}")