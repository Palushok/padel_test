# Models

All models used in the pipeline.

## 1. Player Detection & Tracking

| Parameter | Value |
|-----------|-------|
| Model     | YOLO11n (`yolo11n.pt`) |
| Source    | [Ultralytics](https://docs.ultralytics.com/) — pretrained on COCO |
| Classes   | `person` (class_id=0) + `tennis_racket` (class_id=38) in a single pass |
| Input     | Full frame, 1280px |
| Output    | Bounding boxes `[x1, y1, x2, y2]` + confidence per detection |
| Tracker   | BoT-SORT (built into ultralytics) — applied to persons only |

From each bbox:
- `foot_position`: bottom-center — ground contact point for homography
- `center`: bbox center — fallback for hit proximity
- `crop`: image patch — input for action recognition (if enabled)

**No fine-tuning required.**

### Racket Detection

Rackets are detected in the **same single YOLO inference pass** as players (`classes=[person, tennis_racket]`). Results are split by class: persons get tracked IDs via BoT-SORT, rackets are detection-only (no tracking).

| Parameter | Value |
|-----------|-------|
| Class     | `tennis_racket` (class_id=38) |
| Confidence| 0.3 (default) |

Each detected racket is linked to the nearest player by Euclidean distance between centers. The racket's `owner_track_id` is then used in event detection: ball-to-racket distance takes priority over ball-to-player distance when determining `RACKET_HIT` events.

**No fine-tuning required** — COCO's `tennis_racket` class detects padel rackets adequately.

## 2. Ball Detection

| Parameter | Value |
|-----------|-------|
| Model     | YOLO11n (`yolo11n.pt`) |
| Source    | Ultralytics — pretrained on COCO |
| Class     | `sports_ball` (class_id=32) |
| Confidence| 0.15 (low threshold — ball is small/blurry) |

**Known limitations**: Generic YOLO detects the padel ball inconsistently (30-60% detection rate). The Kalman filter + interpolation compensate.

**Recommended improvements**:
1. Fine-tune YOLO on labeled padel ball data
2. Switch to TrackNet v2/v3 (specialized for racket sport ball tracking)
3. Use higher input resolution (1920+)

## 3. Action Recognition (Optional)

Only used when a trained model is provided. Without it, the pipeline still detects all hits via rule-based approach — just without classifying the hit type.

### Architecture

```
Player Crops (T=16 frames, 112x112)
         │
    ┌────▼────┐
    │  R3D-18 │  (pretrained Kinetics-400)
    └────┬────┘
         │ 512D
    ┌────▼────┐
    │   MLP   │  512 → 256 → 8
    └────┬────┘
         │
    8 action classes
```

| Component | Details |
|-----------|---------|
| Backbone | R3D-18 from `torchvision.models.video`, pretrained Kinetics-400 |
| Input | `(B, 3, T, 112, 112)` player crop sequences |
| Classifier | Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 8) |
| Inference | Sliding window: clip_length=16, stride=8, per tracked player |

### Action Classes

| Class | Description |
|-------|-------------|
| `serve` | Serve motion |
| `return` | Return of serve |
| `volley` | Net volley (no bounce) |
| `groundstroke` | Forehand or backhand from baseline |
| `lob` | High defensive lob |
| `overhead` | Generic overhead shot |
| `smash` | Power overhead smash |
| `wall_play` | Shot played off the wall/glass |

### When disabled vs enabled

| Aspect | Without model | With model |
|--------|--------------|------------|
| Hit detection | Rule-based (ball trajectory + racket/player proximity) | Same rule-based detection |
| Hit type | Not classified (`action_type: null`) | Classified into 8 types |
| Serve detection | Via game FSM (first hit in WAITING_FOR_SERVE) | Via model + FSM confirmation |
| Dependencies | No torch needed for this step | torch + torchvision required |

## 4. Score OCR

| Parameter | Value |
|-----------|-------|
| Model     | EasyOCR |
| Source    | [EasyOCR](https://github.com/JaidedAI/EasyOCR) — open-source |
| Input     | Cropped ROI of score overlay |
| Output    | Text lines → parsed team names + score numbers |
| Languages | Configurable (default: English) |

Reads the broadcast score overlay visible on screen. The score is sampled every N frames (default: 30) rather than every frame for efficiency.

**How it works:**
1. Crop the score region (user-specified ROI or auto-detected bottom-left)
2. Run EasyOCR text detection
3. Cluster text blocks into rows by y-coordinate
4. Parse each row: text = team name, numbers = scores
5. Track score changes between consecutive readings

**No fine-tuning needed** — EasyOCR works well on clean broadcast overlays. For unusual fonts or layouts, the ROI and parsing logic may need adjustment.

## 5. Rally State Classifier

A mandatory component for classifying whether a rally is currently in progress.

### Architecture

| Parameter | Value |
|-----------|-------|
| Backbone | MobileNetV3-Small (`torchvision.models`, pretrained ImageNet) |
| Input | Full frame, downscaled to 224x224 |
| Output | `P(rally_active)` via sigmoid |
| Classifier | Linear(576, 128) → ReLU → Dropout(0.3) → Linear(128, 1) |

### Two modes

| Mode | Source | Training | Accuracy |
|------|--------|----------|----------|
| **Heuristic** (default) | Player tracking movement speeds | Not needed | Baseline |
| **Model** (--rally-model) | MobileNetV3-Small on full frame | Required | Higher |

The heuristic is always active. When a trained model is provided, its predictions override the heuristic on sampled frames and are interpolated across intermediate frames.

### Heuristic approach

Uses player tracking data (already available from Phase 2):
- Computes average movement speed of all tracked players per frame (sliding window of ~30 frames)
- If most players are moving fast → `rally_active`
- If most players are stationary or moving slowly → `rally_inactive`

### Classes

| Class | Description |
|-------|-------------|
| `rally_active` | Ball is in play, players are engaged |
| `rally_inactive` | Between points: players walking, resetting, celebrating |

## 6. Ball Trajectory Filtering (Kalman)

Not a neural network — deterministic `filterpy.KalmanFilter`.

| Parameter | Value |
|-----------|-------|
| State | `[x, y, vx, vy, ax, ay]` |
| Measurement | `[x, y]` |
| Motion model | Constant acceleration |

Smooths trajectory and fills short gaps when ball is undetected.

## Model Sizes

| Model | Size | Download |
|-------|------|----------|
| yolo11n.pt | ~6 MB | Auto-downloaded by ultralytics |
| R3D-18 backbone | ~130 MB | Auto-downloaded by torchvision (only if action recognition enabled) |
| Fine-tuned action model | ~135 MB | User-trained |
| MobileNetV3-Small backbone | ~10 MB | Auto-downloaded by torchvision |
| Fine-tuned rally model | ~12 MB | User-trained |
| EasyOCR models | ~100 MB | Auto-downloaded on first run |
