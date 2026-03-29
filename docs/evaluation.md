# Evaluation

How to measure quality of each pipeline component and the pipeline as a whole.

## Quick Start

```bash
# Run the pipeline
python scripts/run_pipeline.py video.mp4 \
  --floor-points '...' --wall-top-points '...' \
  -j predictions.json

# Evaluate against ground truth
python scripts/evaluate.py predictions.json ground_truth.json

# With custom thresholds + save report
python scripts/evaluate.py predictions.json ground_truth.json \
  --event-tolerance 10 \
  --ball-threshold 20 \
  -o report.json -v
```

The evaluator automatically detects which sections are present in **both** the prediction and ground truth files and evaluates only those sections.

---

## Ground Truth Format

The ground truth JSON uses the **same schema** as the pipeline output (see `docs/data_format.md`). You only need to include the sections you want to evaluate.

### Minimal example — events only

```json
{
  "low_level_events": [
    {"type": "racket_hit",   "frame_idx": 100},
    {"type": "floor_bounce", "frame_idx": 120},
    {"type": "wall_bounce",  "frame_idx": 145},
    {"type": "racket_hit",   "frame_idx": 165}
  ],
  "high_level_events": [
    {"type": "serve",     "frame_idx": 100},
    {"type": "point_won", "frame_idx": 220, "details": {"winning_team": 1}}
  ]
}
```

### Full example — all components

```json
{
  "ball_tracking": [
    {"frame_idx": 0, "position": [500.2, 300.1], "visible": true},
    {"frame_idx": 1, "position": null,           "visible": false},
    {"frame_idx": 2, "position": [502.5, 298.0], "visible": true}
  ],

  "player_tracking": [
    {
      "frame_idx": 0,
      "players": [
        {"track_id": 1, "bbox": [100, 200, 150, 350]},
        {"track_id": 2, "bbox": [300, 210, 360, 360]},
        {"track_id": 3, "bbox": [500, 500, 560, 680]},
        {"track_id": 4, "bbox": [700, 490, 760, 670]}
      ]
    }
  ],

  "low_level_events": [
    {"type": "racket_hit",   "frame_idx": 100},
    {"type": "floor_bounce", "frame_idx": 120}
  ],

  "high_level_events": [
    {"type": "serve",     "frame_idx": 100},
    {"type": "point_won", "frame_idx": 220, "details": {"winning_team": 1}}
  ],

  "rally_states": [
    {"frame_idx": 0,   "is_rally_active": false},
    {"frame_idx": 50,  "is_rally_active": false},
    {"frame_idx": 100, "is_rally_active": true},
    {"frame_idx": 200, "is_rally_active": true},
    {"frame_idx": 250, "is_rally_active": false}
  ],

  "ocr_readings": [
    {
      "frame_idx": 0,
      "team_a_name": "TAPIA / CHINGOTTO",
      "team_b_name": "LEBRON / GALAN",
      "team_a_scores": [0],
      "team_b_scores": [0]
    }
  ]
}
```

### How to create ground truth

1. **Bootstrap from predictions**: run the pipeline, then manually correct the output JSON.
2. **Manual annotation**: use a video labeling tool (e.g. CVAT, VIA) to annotate ball positions and events, then convert to our JSON format.
3. **Partial GT**: you don't have to annotate everything — include only the sections you care about.

---

## Metrics Reference

### 1. Ball Detection (`src/evaluation/ball.py`)

| Metric | Description |
|--------|-------------|
| **Detection rate** | Fraction of GT-visible frames where the ball was detected within threshold |
| **Precision** | Correct detections / total detections |
| **Recall** | Correct detections / total GT-visible frames |
| **F1** | Harmonic mean of precision and recall |
| **Position error** | Mean, median, P90, std of pixel distance between predicted and GT centres |

A detection is a **true positive** if the predicted center is within `--ball-threshold` pixels (default: 15) of the GT center.

### 2. Player Tracking (`src/evaluation/tracking.py`)

| Metric | Description |
|--------|-------------|
| **HOTA** | Higher Order Tracking Accuracy: `sqrt(DetA × AssA)` averaged over IoU thresholds 0.05–0.95 |
| **DetA** | Detection Accuracy: `TP / (TP + FP + FN)` — how well the tracker finds all objects |
| **AssA** | Association Accuracy: average track-level IoU — how well identities are linked over time |
| **LocA** | Localization Accuracy: average IoU of matched detection pairs |
| **MOTA** | Multi-Object Tracking Accuracy: `1 − (FN + FP + ID_switches) / GT_boxes` |
| **MOTP** | Multi-Object Tracking Precision: mean position error of matched boxes (px) |
| **Detection rate** | Fraction of GT boxes matched by a prediction |
| **ID switches** | Number of frames where a GT identity got a different predicted track ID |
| **Position error** | Mean, median, P90, std of pixel distance between matched bbox centres |

HOTA is computed over 19 IoU thresholds (0.05 to 0.95) and decomposes cleanly into detection quality (DetA) and association quality (AssA), making it easier to diagnose whether the tracker struggles with finding objects or maintaining identities. MOTA/MOTP use a single IoU threshold (`--iou-threshold`, default: 0.3).

### 3. Low-Level Events (`src/evaluation/events.py`)

| Metric | Description |
|--------|-------------|
| **Per-type P/R/F1** | For each event type (floor_bounce, wall_bounce, racket_hit, net_hit) |
| **Aggregate P/R/F1** | Over all types combined |
| **Temporal error** | Mean/median/max frame distance between matched pred↔GT events |

A predicted event is a **true positive** if there is a GT event of the same type within `--event-tolerance` frames (default: 5). Greedy matching by ascending frame distance.

### 4. High-Level Events (`src/evaluation/events.py`)

Same methodology as low-level events, applied to high-level event types (serve, point_won, game_over, set_over, etc.).

### 5. Rally State (`src/evaluation/rally.py`)

| Metric | Description |
|--------|-------------|
| **Accuracy** | Fraction of frames with correct rally_active/inactive classification |
| **rally_active P/R/F1** | TP = rally correctly detected, FP = predicted active but GT inactive, FN = GT active but predicted inactive |
| **rally_inactive P/R/F1** | TP = pause correctly detected, FP = predicted inactive but GT active, FN = GT inactive but predicted active |

Evaluated on frames present in both prediction and GT. Per-class FP/FN are symmetric: `active.FP == inactive.FN` and vice versa.

### 6. OCR Score (`src/evaluation/score.py`)

| Metric | Description |
|--------|-------------|
| **Score accuracy** | Fraction of matched readings where the score is identical |
| **Name accuracy** | Fraction of matched readings where team names are identical |
| **Matched readings** | How many GT readings found a prediction within `--ocr-tolerance` frames |

### 7. Final Score (`src/evaluation/score.py`)

Counts `point_won` events per team in predictions and GT. Reports whether the total points per team match.

---

## CLI Options

```
python scripts/evaluate.py PREDICTIONS GROUND_TRUTH [OPTIONS]

Arguments:
  PREDICTIONS    Pipeline output JSON
  GROUND_TRUTH   Ground truth annotations JSON

Options:
  -t, --event-tolerance INT    Temporal tolerance for events (frames)  [default: 5]
  -b, --ball-threshold FLOAT   Max pixel distance for ball TP          [default: 15]
      --iou-threshold FLOAT    Min IoU for tracking match              [default: 0.3]
      --ocr-tolerance INT      Temporal tolerance for OCR matching     [default: 30]
  -o, --output-json PATH       Save full report as JSON
  -v, --verbose                Debug logging
```

## Output

### Terminal

```
================================================================
EVALUATION REPORT
================================================================
  ball                   det_rate=45.2%  P=89.1%  R=45.2%  F1=59.9%  mean_err=6.3px
  tracking               HOTA=82.1%  DetA=90.5%  AssA=74.5%  MOTA=78.3%  id_sw=3
  low_level_events       P=72.0%  R=68.5%  F1=70.2%  (tol=5f)
  high_level_events      P=85.0%  R=80.0%  F1=82.4%  (tol=5f)
  rally_state            acc=88.5%  P=91.2%  R=86.3%  F1=88.7%
  ocr_score              score_acc=95.0%  matched=19/20
  final_score            MATCH
================================================================
```

### JSON (`--output-json`)

Full report with all per-type breakdowns, error distributions, and matched/unmatched details.

---

## Programmatic Usage

```python
import json
from src.evaluation import evaluate_all

with open("predictions.json") as f:
    pred = json.load(f)
with open("ground_truth.json") as f:
    gt = json.load(f)

report = evaluate_all(pred, gt, event_tolerance=10)

print(report["summary"])
print(report["low_level_events"]["per_type"])
```

---

## Tuning Workflow

1. Run the pipeline on a test video.
2. Create ground truth annotations (manually or by correcting the pipeline output).
3. Run evaluation → identify weak points.
4. Tune parameters (see `docs/training.md` § Parameter Tuning) or fine-tune models.
5. Re-run pipeline → re-evaluate → iterate.

### What to improve based on metrics

| Low metric | Likely cause | Action |
|------------|-------------|--------|
| Ball detection rate low | YOLO misses small ball | Fine-tune YOLO on padel data or switch to TrackNet |
| Ball position error high | Detection offset | Increase input resolution (`imgsz`) |
| HOTA low, DetA low | Tracker misses objects or has many FP | Tighten detection confidence, filter by court ROI |
| HOTA low, AssA low | Tracker loses identity over time | Tune BoT-SORT parameters (track buffer, matching threshold) |
| MOTA low, many FP | Spectators detected as players | Tighten detection confidence or filter by court ROI |
| MOTA low, many ID switches | Tracker loses identity | Tune BoT-SORT parameters |
| Low-level event F1 low | Wrong bounce classification | Tune `velocity_change_threshold`, trajectory analysis weights |
| High-level event F1 low | Game logic errors | Check FSM rules, increase event tolerance |
| Rally accuracy low | Poor speed threshold | Tune `rally_classifier.heuristic.speed_threshold` or train model |
| OCR score accuracy low | Wrong ROI / noisy text | Adjust `--score-roi`, increase OCR resolution |
| Final score mismatch | Accumulated errors | Fix underlying event/OCR issues |
