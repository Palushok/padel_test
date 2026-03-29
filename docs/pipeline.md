# Pipeline

Detailed description of each pipeline phase and its components.

## Phase 1: Initialization

Runs once on the first frame of the video.

### Court Detection (`src/court/detector.py`)

Two modes:
- **manual**: User provides 8 pixel coordinates in the config (or via CLI flags).
- **auto**: Floor corners detected via Canny + Hough lines. Wall top corners still require manual input.

**8-point marking order**:
```
Points 0-3 (floor):     far-left, far-right, near-right, near-left
Points 4-7 (wall tops): far-left-top, far-right-top, near-right-top, near-left-top
```

"Far" = further from camera (top of image), "Near" = closer to camera (bottom of image).

### Homography (`src/court/homography.py`)

Computes a perspective transform from 4 floor corners (pixels) to real-world coordinates (meters) using `cv2.findHomography`. Standard padel court: 10m wide x 20m long.

Provides:
- `image_to_court(pixel_point) → (x_meters, y_meters)`
- `court_to_image(court_point) → (pixel_x, pixel_y)`
- `get_court_side(court_point) → "near" | "far"`

### Court Geometry (`src/court/geometry.py`)

Builds 5 surface polygons from the 8 marked points:

| Surface      | Point indices | Description              |
|-------------|---------------|--------------------------|
| `FLOOR`     | [0,1,2,3]     | Playing surface          |
| `WALL_FAR`  | [4,5,1,0]     | Back wall (far from cam) |
| `WALL_RIGHT`| [5,6,2,1]     | Right side wall          |
| `WALL_NEAR` | [6,7,3,2]     | Back wall (near to cam)  |
| `WALL_LEFT` | [7,4,0,3]     | Left side wall           |

Key method: `classify_by_court_coords(court_point, image_point) → Surface` — uses the floor homography to determine surface. A ball on the floor maps to court coordinates within bounds; a ball on a wall (at height Z>0) maps outside bounds. The overflow direction indicates the wall. Falls back to image-space polygon test in the ambiguous boundary zone.

---

## Phase 2: Frame-by-Frame Processing (Single Video Pass)

Runs on every frame sequentially in a single `cv2.VideoCapture` loop. All per-frame work happens here — no additional video passes.

### Player & Racket Tracking (`src/players/tracker.py`)

Single YOLO inference per frame with `classes=[person, tennis_racket]` + BoT-SORT tracking. Results are split by class: persons get tracked IDs, rackets are detection-only.

Per frame produces `FrameTracking`:
- `TrackedPlayer`: `track_id`, `bbox`, `foot_position` (bottom-center), `center`, `confidence`, `crop`.
- `DetectedRacket`: `bbox`, `center`, `confidence`, `owner_track_id` (nearest player).

### Ball Detection (`src/ball/detector.py`)

YOLO11 with class `sports_ball` (class_id=32). Returns highest-confidence detection or `None`.

### OCR Score Sampling (`src/score/ocr.py`)

Every `sample_interval` frames (default: 30), the current frame is passed to `ScoreOCR.read_frame()` to read the broadcast score overlay. Results are collected for Phase 3 reconciliation.

### Rally State Sampling (`src/players/rally_classifier.py`)

When a trained rally model is provided (`--rally-model`), every `sample_interval` frames (default: 5), the current frame is passed through MobileNetV3-Small to classify whether a rally is in progress. Results are collected for Phase 3 interpolation and event interpretation.

---

## Phase 3: Post-Processing

Runs after all frames have been processed.

### 3a. Team Assignment (`src/players/team_assigner.py`)

Position-based: for each unique `track_id`, median Y-coordinate on the court (via homography). Far half → team 0, near half → team 1.

**Side switching**: After every odd total game count (1, 3, 5, ...), teams switch sides. The FSM emits `SIDE_SWITCH` events.

### 3b. Ball Interpolation (`src/ball/interpolator.py`)

Fills gaps using `scipy.interpolate.interp1d` (cubic or linear). Max gap: 10 frames (configurable). Runs **before** Kalman — takes advantage of offline processing (can see future positions to fill gaps accurately).

### 3c. Ball Kalman Filter (`src/ball/kalman_filter.py`)

Constant-acceleration model: state `[x, y, vx, vy, ax, ay]`.
Receives the merged trajectory (detected + interpolated positions). Interpolated positions are fed with higher measurement noise (`interpolated_noise_scale`, default 3x) so the filter trusts them less than real detections.

Outputs `FilteredBallState`: position, velocity, acceleration, `is_observed`, `is_detected`.

### 3d. Action Recognition — OPTIONAL (`src/players/action_recognition.py`)

Only runs when a trained model is provided (`--action-model` or config `action_recognition.enabled: true`).

Sliding window (16 frames, stride 8) over each player's track. Player crops are resized to 112x112 and fed through R3D-18 → MLP → 8 action classes.

**When enabled**: `RACKET_HIT` events get an `action_type` field (serve, return, volley, etc.)

**When disabled**: Hits are still detected (rule-based), but `action_type` is `None`. The serve is identified by the FSM (first hit during `WAITING_FOR_SERVE`).

### 3e. Rally State Classification (`src/players/rally_classifier.py`)

Mandatory step. Produces a `RallyState` for every frame indicating whether a rally is in progress.

**Heuristic** (always active): Computes average player movement speed from tracking data using a sliding window (~30 frames). Players moving fast → `rally_active`. Players stationary → `rally_inactive`.

**Model** (when weights provided): MobileNetV3-Small predictions from Phase 2 are interpolated to cover all frames. Model predictions override the heuristic where available.

Output: `list[RallyState]` — one per frame, with `is_rally_active`, `confidence`, and `source` ("heuristic", "model", or "model_interpolated").

### 3f. Trajectory Analysis (`src/ball/trajectory_analyzer.py`)

Instantiated when `trajectory_analysis.enabled: true` (default). Used by the low-level event detector to resolve ambiguous floor/wall bounces in the near-wall overlap zone.

Combines four signals to compute `P(floor)`:
1. **Rules prior**: event sequence since last RACKET_HIT (padel mandates floor-first contact)
2. **Court-Y displacement**: post-bounce horizontal movement in court coordinates
3. **Image curvature**: quadratic fit to post-bounce image-space trajectory
4. **Rally state**: rally continuity from the RallyClassifier

Confidence score stored on each disambiguated event for downstream filtering.

### 3g. Low-Level Event Detection (`src/events/low_level.py`)

Finds frames where ball velocity/direction changes sharply. Each change point is classified:

1. **RACKET_HIT**: Ball within `racket_hit_distance` of a detected racket, or within `player_hit_distance` of a player (fallback)
2. **NET_HIT**: Ball in net zone and speed drops significantly
3. **FLOOR_BOUNCE**: Ball on `FLOOR` surface (via `CourtGeometry`)
4. **WALL_BOUNCE**: Ball on any wall surface (with specific wall in `surface` field)

When the ball is in the **ambiguous boundary zone** (where `CourtGeometry.is_ambiguous_zone()` returns `True`), classification is deferred to the `TrajectoryAnalyzer`, which combines padel rules, trajectory shape, and rally state to disambiguate. The resulting `confidence` score is stored on the event.

If action recognition is enabled, `RACKET_HIT` events include the `action_type`.

### 3h. High-Level Game FSM (`src/events/high_level.py`)

State machine: `WAITING_FOR_SERVE → SERVE_IN_PROGRESS → RALLY → POINT_SCORED`

Key transitions:
- **RACKET_HIT** during `WAITING_FOR_SERVE` → `SERVE` event
- **FLOOR_BOUNCE** on opponent's side during serve → enter `RALLY`
- **Double floor bounce** on same side → `POINT_WON`
- **NET_HIT** during rally → `POINT_WON` for opponent
- **WALL_BOUNCE** on hitting team's side → `POINT_WON` for opponent
- 2 serve faults → double fault → `POINT_WON`

Scoring: 0/15/30/40, deuce, advantage, games, sets, best-of-3. Side switch after odd game totals.

### 3i. OCR Score Reconciliation (`src/score/reconciler.py`)

OCR readings were already collected during Phase 2 (every `sample_interval` frames). This step reconciles them with FSM events.

#### ScoreReconciler

Compares FSM-predicted score changes with OCR-observed score changes. **OCR is treated as ground truth.**

Algorithm:
1. Detect score changes from consecutive OCR readings (score went up for one team)
2. Match each OCR change to the nearest FSM `POINT_WON` event (same team). The scoreboard typically updates *after* the real event, so we allow the FSM event to precede the OCR change by up to `confirmation_delay_sec` seconds (default: 3.0).
3. **Matched**: FSM point confirmed, kept as-is
4. **Unmatched FSM point** (FSM says point, OCR disagrees within the delay window): removed along with cascade events (game_over, etc.)
5. **Unmatched OCR change** (OCR shows score change, FSM has no point): new `POINT_WON` inserted with `reason: "ocr_detected"`

This ensures the output score matches the actual broadcast score even when ball detection or event rules make errors.

---

## Phase 4: Visualization (`src/visualization/renderer.py`)

Output video with:
- Court outline (green) + net line (white)
- Player bboxes colored by team (blue/red)
- Ball (yellow circle) with trailing path
- Event markers with labels (action type shown when available)
- Score bar at bottom
- Bird's-eye minimap in corner

---

## CLI Usage

```bash
# Basic (rule-based hit detection, no action classification):
python scripts/run_pipeline.py video.mp4 \
  --floor-points '[[200,150],[800,150],[900,650],[100,650]]' \
  --wall-top-points '[[180,50],[820,50],[950,20],[80,20]]' \
  -o output.mp4 -j results.json -v

# With action recognition model:
python scripts/run_pipeline.py video.mp4 \
  --floor-points '[[200,150],[800,150],[900,650],[100,650]]' \
  --wall-top-points '[[180,50],[820,50],[950,20],[80,20]]' \
  --action-model weights/action_model.pt \
  -o output.mp4 -j results.json

# With custom score overlay region:
python scripts/run_pipeline.py video.mp4 \
  --floor-points '...' --wall-top-points '...' \
  --score-roi '[50, 900, 500, 1000]' \
  -o output.mp4 -j results.json

# With rally classifier model:
python scripts/run_pipeline.py video.mp4 \
  --floor-points '...' --wall-top-points '...' \
  --rally-model weights/rally_model.pt \
  -o output.mp4 -j results.json

# Full pipeline (action recognition + rally model + custom score ROI):
python scripts/run_pipeline.py video.mp4 \
  --floor-points '...' --wall-top-points '...' \
  --action-model weights/action_model.pt \
  --rally-model weights/rally_model.pt \
  --score-roi '[50, 900, 500, 1000]' \
  -o output.mp4 -j results.json -v

# Evaluate results against ground truth annotations:
python scripts/evaluate.py results.json ground_truth.json -o report.json
```

See `docs/evaluation.md` for ground truth annotation format and metrics.
