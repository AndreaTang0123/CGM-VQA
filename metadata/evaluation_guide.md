# CGM-VQA Evaluation Guide

This document describes how to use the evaluation dataset to assess a model's ability to interpret CGM (Continuous Glucose Monitor) graphs.

---

## Dataset Overview

| Property | Value |
|---|---|
| Total samples | 100 |
| Graphs | 25 (one per day) |
| Questions per graph | 4 (2 yes/no + 2 temporal) |
| Question pool | 40 questions (Q01–Q40) |
| Image format | PNG (cropped daily CGM plots) |

---

## File Structure

```
CGM-VQA-1/
├── graphs_cropped/          ← CGM graph images (PNG)
├── metadata/
│   ├── graph_metadata.csv   ← graph_id ↔ filename mapping
│   ├── question_pool.csv    ← all 40 questions with type & category
│   └── eval_questions.json  ← evaluation prompts (no answers)
└── annotation/
    └── annotations.json     ← ground-truth answers
```

---

## Task Types

### `yes_no`
Questions that require a **yes** or **no** answer about the CGM graph (e.g., whether glucose exceeded a threshold, whether a trend was observed).

**Model output format:** `yes` or `no`

### `temporal`
Questions that ask the model to **identify a time period or interval** from the graph (event localization, trend segments, duration segments).

**Model output format:** a time string such as `HH:MM` for a point or `HH:MM-HH:MM` for an interval.

---

## Input Format (`eval_questions.json`)

Each entry contains:

```json
{
  "sample_id": "S001",
  "graph_id": "G01",
  "image_file": "Readings_2023-12-09.png",
  "image_path": "graphs_cropped/Readings_2023-12-09.png",
  "question_id": "Q17",
  "task_type": "yes_no",
  "question": "Did the largest meal event lead to a glucose spike?",
  "expected_answer_format": "yes or no"
}
```

> **Note:** `eval_questions.json` does **not** include ground-truth answers. Answers are stored separately in `annotation/annotations.json`.

---

## Suggested Prompt Template

For multimodal models (e.g., GPT-4o, Gemini, Claude):

```
You are analyzing a Continuous Glucose Monitor (CGM) graph for a single day.
The x-axis shows time (00:00–23:59) and the y-axis shows glucose level in mg/dL.
The red dashed line marks 180 mg/dL (hyperglycemia threshold).
The blue dashed line marks 70 mg/dL (hypoglycemia threshold).

Question: {question}

Answer format: {expected_answer_format}
Answer:
```

---

## Evaluation Metrics

| Task Type | Recommended Metric |
|---|---|
| `yes_no` | Accuracy (exact match) |
| `temporal` | IoU (Intersection over Union) of predicted vs. ground-truth time interval; threshold ≥ 0.5 for correct |

### IoU for Temporal Questions

Given a predicted interval `[p_start, p_end]` and ground-truth `[t_start, t_end]`:

```
intersection = max(0, min(p_end, t_end) - max(p_start, t_start))
union        = max(p_end, t_end) - min(p_start, t_start)
IoU          = intersection / union
```

A prediction is considered **correct** if `IoU ≥ 0.5`.

---

## Question Categories

| Category | Question IDs | Count |
|---|---|---|
| `point_reading` | Q01–Q04 | 4 |
| `trend_understanding` | Q05–Q08 | 4 |
| `duration_reasoning` | Q09–Q12 | 4 |
| `comparison` | Q13–Q16 | 4 |
| `event_reasoning` | Q17–Q20 | 4 |
| `event_localization` | Q21–Q25 | 5 |
| `trend_segments` | Q26–Q30 | 5 |
| `event_related_segments` | Q31–Q35 | 5 |
| `duration_segments` | Q36–Q40 | 5 |
