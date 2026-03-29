# Architecture

## Overview

The padel analysis pipeline processes a recorded video of a padel match and produces:
1. **Player & racket tracking** — per-frame positions of all 4 players and their rackets on the court
2. **Events** — two-level event system (low-level physical events + high-level game events)
3. **Score tracking** — OCR-based reading of the broadcast score overlay, used as ground truth
4. **Rally state classification** — per-frame detection of whether a rally is in progress
5. **(Optional) Hit type classification** — serve, return, volley, groundstroke, lob, overhead, smash, wall_play

The system is designed for **offline (batch) processing**: the entire video is analyzed first, then events are interpreted from the collected predictions.

## Assumptions

- **Static camera** mounted behind and above the court, viewing the entire playing area
- **Full court visible** including all walls, glass, net, and floor
- **4 players** (2v2 padel doubles)
- Homography is computed once on the first frame and reused throughout

## Project Structure

```
padel/
├── config/
│   └── default.yaml              # All pipeline parameters
├── docs/                         # Documentation
├── scripts/
│   └── run_pipeline.py           # CLI entry point
└── src/
    ├── pipeline.py               # Main orchestrator
    ├── court/
    │   ├── detector.py           # Court corner detection (manual / auto)
    │   ├── geometry.py           # 3D court geometry (5 surfaces from 8 points)
    │   └── homography.py         # Floor homography (image ↔ court meters)
    ├── players/
    │   ├── detector.py           # YOLO person detection
    │   ├── tracker.py            # BoT-SORT player tracking + racket detection
    │   ├── team_assigner.py      # Position-based team assignment + side switching
    │   ├── rally_classifier.py   # Rally state classifier (MobileNetV3 + heuristic)
    │   └── action_recognition.py # (Optional) R3D-18 video action classifier
    ├── ball/
    │   ├── detector.py           # YOLO ball detection
    │   ├── kalman_filter.py      # Constant-acceleration Kalman filter
    │   ├── interpolator.py       # Cubic/linear interpolation for missing frames
    │   └── trajectory_analyzer.py # Trajectory-based floor/wall disambiguation
    ├── events/
    │   ├── low_level.py          # Physical events (bounces, hits, net)
    │   └── high_level.py         # Game FSM (serve → rally → point → game → set)
    ├── score/
    │   ├── ocr.py                # OCR score reader (EasyOCR)
    │   └── reconciler.py         # Reconciles OCR score with FSM events
    ├── evaluation/
    │   ├── metrics.py            # Generic helpers (P/R/F1, temporal matching)
    │   ├── ball.py               # Ball detection evaluation
    │   ├── tracking.py           # Player tracking evaluation (MOTA/MOTP)
    │   ├── events.py             # Event detection evaluation (low + high level)
    │   ├── rally.py              # Rally state classification evaluation
    │   ├── score.py              # OCR / score accuracy evaluation
    │   └── report.py             # Unified evaluation runner
    └── visualization/
        └── renderer.py           # Video overlay + bird's-eye minimap
```

## Data Flow

```
Video file
  │
  ├─ Phase 1: INITIALIZATION (first frame only)
  │    ├─ CourtDetector → 8 image points (4 floor + 4 wall tops)
  │    ├─ HomographyTransformer.compute(floor_corners) → 3x3 matrix
  │    └─ CourtGeometry.build(all_8_points) → 5 surface polygons
  │
  ├─ Phase 2: FRAME-BY-FRAME (single video pass)
  │    ├─ PlayerTracker.track_frame() → FrameTracking (players + rackets)
  │    ├─ BallDetector.detect() → BallDetection | None
  │    ├─ ScoreOCR.read_frame() → ScoreReading (every N frames)
  │    └─ RallyClassifier.predict_frame() → RallyState (model, every M frames)
  │
  └─ Phase 3: POST-PROCESSING (after all frames)
       ├─ TeamAssigner.assign_teams() → {track_id: team_id}
       ├─ BallInterpolator.interpolate() → fill gaps (uses future data)
       ├─ BallKalmanFilter.filter_trajectory() → smooth positions, velocity, acceleration
       ├─ (Optional) ActionRecognizer.predict_actions() → hit type per window
       ├─ RallyClassifier.classify_all() → per-frame rally state (heuristic + model)
       ├─ TrajectoryAnalyzer → disambiguate floor/wall in overlap zones
       ├─ LowLevelEventDetector.detect() → floor/wall bounces, racket hits, net hits
       ├─ GameStateMachine.process() → serves, points, games, sets, side switches
       └─ ScoreReconciler.reconcile() → correct FSM using OCR readings
```

## Racket-Based Hit Detection

The pipeline detects rackets alongside players using the same YOLO model (COCO class `tennis_racket`). Each detected racket is associated with the nearest player. When the ball changes velocity sharply, the event detector first checks **ball-to-racket distance**. If no racket is close enough, it falls back to **ball-to-player distance** (larger threshold). This two-tier approach produces fewer false positives compared to player-only proximity.

### Without action recognition model (default)

Hits are detected rule-based via racket/player proximity. No hit type classification (all hits are generic). The **serve** is still identified by the game state machine (first hit in `WAITING_FOR_SERVE` state → serve).

### With action recognition model (--action-model)

In addition to rule-based detection, player crop sequences are classified by a trained R3D-18 model into specific hit types: serve, return, volley, groundstroke, lob, overhead, smash, wall_play. The `action_type` field on `RACKET_HIT` events is populated.

## Score OCR & Reconciliation

The broadcast video displays the current score as an on-screen overlay (team names + score numbers). The pipeline always reads this overlay via **EasyOCR** and uses it as the **authoritative source of truth** for scoring.

### Why this matters

Ball detection and event rules are imperfect. A missed ball detection can cause the FSM to miss a point, or a noisy trajectory can trigger a false double-bounce. The broadcast score, however, is always correct — it reflects the official referee decisions.

### How it works

1. **ScoreOCR** (`src/score/ocr.py`) reads the score overlay region every N frames (default: 30). It parses team names and score numbers from two text rows.

2. **ScoreReconciler** (`src/score/reconciler.py`) compares OCR-observed score changes with FSM-predicted `POINT_WON` events:

   | OCR | FSM | Result |
   |-----|-----|--------|
   | Score changed for team A | `POINT_WON` for team A nearby | **Confirmed** — event kept |
   | No score change | `POINT_WON` predicted | **False positive** — event + cascade removed |
   | Score changed | No `POINT_WON` found | **Missed event** — `POINT_WON` inserted with `reason: "ocr_detected"` |

3. The reconciled event list replaces the original FSM output. All non-scoring events (serves, hits, bounces) are preserved as-is.

### Configuration

- `--score-roi '[x1, y1, x2, y2]'` — pixel region of the score overlay (or auto-detect)
- `score_ocr.sample_interval` — how often to read (every N frames)
- `score_ocr.reconciliation.confirmation_delay_sec` — how many seconds to wait for the scoreboard to update after an event (default: 3.0)

## Trajectory-Based Surface Disambiguation

The near-wall overlap zone presents a fundamental monocular depth ambiguity: a ball on the floor near the near wall and a ball hitting the near wall at ~1m height can project to the same image pixel. The `TrajectoryAnalyzer` (`src/ball/trajectory_analyzer.py`) resolves this by combining four independent signals:

1. **Padel rule prior**: After a `RACKET_HIT`, the ball must bounce on the floor first before contacting any wall. The event sequence strongly constrains classification.
2. **Post-bounce court-Y displacement**: Wall bounces reverse the ball's horizontal direction (large court-Y change), while floor bounces send the ball upward (smaller court-Y change initially).
3. **Image-space curvature**: Floor bounces produce a pronounced parabolic arc; wall bounces produce a flatter post-bounce trajectory.
4. **Rally state**: If the rally is active and this is the first contact after a hit, the contact is likely a legal floor bounce.

Each signal produces a `P(floor)` value; they are combined via weighted average (weights configurable in `trajectory_analysis.weights`). The `LowLevelEventDetector` uses this analyzer only when `CourtGeometry.is_ambiguous_zone()` returns `True`, preserving the faster homography-based classification for unambiguous regions.

## Rally State Classifier

A **mandatory** component (`src/players/rally_classifier.py`) that determines whether a rally is in progress on each frame.

### Why it matters

Rally state provides critical context for event interpretation. If the rally is active, contacts are legal (ball was in-bounds). If players relax and stop moving, the rally ended — indicating a fault or out.

### How it works

**Heuristic mode** (always active): Computes average player movement speed from tracking data using a sliding window. Fast-moving players indicate an active rally.

**Model mode** (when `--rally-model` is provided): MobileNetV3-Small binary classifier on the full frame. Runs every N frames during Phase 2; results are interpolated across intermediate frames. Model predictions override the heuristic where available.

### Configuration

- `--rally-model path/to/weights.pt` — path to trained rally classifier weights
- `rally_classifier.sample_interval` — how often to run model inference (default: 5 frames)
- `rally_classifier.heuristic.speed_threshold` — player speed below which they are "stationary"

---

## Key Design Decisions

1. **Offline-first**: All detection/tracking happens in Phase 2 (single video pass) before any event logic runs. This lets the interpolator look at future ball positions and the Kalman filter operate on a denser trajectory.

2. **Court geometry via homography + trajectory analysis**: 8 user-marked points define the court. Surface classification uses the floor homography: a ball on the floor maps to coordinates within [0,10]×[0,20]m, while a ball on a wall (Z>0) maps outside these bounds. In the ambiguous boundary zone (where floor and wall projections overlap due to monocular depth), the `TrajectoryAnalyzer` takes over, using padel rules, post-bounce displacement, curvature, and rally state to disambiguate.

3. **Team assignment by position, not appearance**: Players always stay on their half. Median court-Y per track assigns teams. The FSM handles side switching after odd game totals.

4. **Action recognition is optional**: The pipeline works fully without it — rule-based hit detection + the FSM still produce all game events. The action model adds granularity to hit types when available.

5. **Event hierarchy**: Low-level events are physical (ball changed direction near surface X). High-level events apply padel rules (double bounce = point, etc.).

6. **OCR as ground truth**: The on-screen broadcast score is always read and treated as the authoritative source. If the FSM predicts a point that OCR doesn't confirm, the point is removed. If OCR shows a score change the FSM missed, a point is inserted. This ensures the output matches the actual match result even when ball/event detection is imperfect.
