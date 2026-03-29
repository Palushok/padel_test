# Data Format

## Input

### Video File
Any format supported by OpenCV (mp4, avi, mov, mkv). Static camera, overhead behind-the-court angle, full court visible.

### Court Calibration (8 points)

In config or via CLI. All coordinates in pixels `[x, y]`.

```yaml
court:
  manual_points:
    floor:
      - [200, 150]   # 0: far-left
      - [800, 150]   # 1: far-right
      - [900, 650]   # 2: near-right
      - [100, 650]   # 3: near-left
    wall_tops:
      - [180, 50]    # 4: far-left-top
      - [820, 50]    # 5: far-right-top
      - [950, 20]    # 6: near-right-top
      - [80, 20]     # 7: near-left-top
```

CLI equivalent:
```bash
--floor-points '[[200,150],[800,150],[900,650],[100,650]]'
--wall-top-points '[[180,50],[820,50],[950,20],[80,20]]'
```

### Real-World Dimensions (hardcoded)
- Court: 20m x 10m
- Wall height: 4m
- Net at Y = 10m (center)

---

## Output JSON (`--output-json`)

```json
{
  "metadata": {
    "total_frames": 18000,
    "fps": 30.0,
    "width": 1920,
    "height": 1080
  },

  "team_assignments": {
    "1": 0, "2": 0, "3": 1, "4": 1
  },

  "player_tracking": [
    {
      "frame_idx": 0,
      "players": [
        {
          "track_id": 1,
          "bbox": [100.5, 200.3, 150.8, 350.1],
          "foot_position": [125.65, 350.1],
          "team_id": 0,
          "confidence": 0.92
        }
      ],
      "rackets": [
        {
          "bbox": [140.0, 220.0, 165.0, 280.0],
          "center": [152.5, 250.0],
          "confidence": 0.78,
          "owner_track_id": 1
        }
      ]
    }
  ],

  "ball_tracking": [
    {
      "frame_idx": 0,
      "position": [500.2, 300.1],
      "velocity": [5.3, -2.1],
      "is_observed": true
    }
  ],

  "action_predictions": [
    {
      "track_id": 1,
      "action": "groundstroke",
      "confidence": 0.85,
      "frame_start": 100,
      "frame_end": 115,
      "probabilities": {
        "serve": 0.02, "return": 0.05, "volley": 0.01,
        "groundstroke": 0.85, "lob": 0.02, "overhead": 0.02,
        "smash": 0.01, "wall_play": 0.02
      }
    }
  ],

  "low_level_events": [
    {
      "type": "racket_hit",
      "frame_idx": 165,
      "ball_position_image": [320.1, 400.8],
      "ball_position_court": [3.8, 14.2],
      "player_track_id": 3,
      "court_side": "near",
      "surface": null,
      "action_type": "groundstroke"
    },
    {
      "type": "floor_bounce",
      "frame_idx": 150,
      "ball_position_image": [450.2, 380.5],
      "ball_position_court": [5.1, 12.3],
      "player_track_id": null,
      "court_side": "near",
      "surface": "floor",
      "action_type": null,
      "confidence": 0.82
    },
    {
      "type": "wall_bounce",
      "frame_idx": 180,
      "ball_position_image": [50.0, 300.0],
      "ball_position_court": [0.5, 8.0],
      "player_track_id": null,
      "court_side": "far",
      "surface": "wall_left",
      "action_type": null
    }
  ],

  "high_level_events": [
    {
      "type": "serve",
      "frame_idx": 100,
      "details": {"serving_team": 0, "player_id": 1}
    },
    {
      "type": "point_won",
      "frame_idx": 220,
      "details": {
        "winning_team": 1,
        "reason": "double_bounce",
        "score_before": "Sets: [] Game: 0-0 Points: 0-0"
      }
    },
    {
      "type": "side_switch",
      "frame_idx": 5000,
      "details": {"new_sides": {"0": "near", "1": "far"}, "total_games": 1}
    }
  ],

  "rally_states": [
    {
      "frame_idx": 0,
      "is_rally_active": false,
      "confidence": 0.85,
      "source": "heuristic"
    },
    {
      "frame_idx": 100,
      "is_rally_active": true,
      "confidence": 0.92,
      "source": "model"
    }
  ],

  "ocr_readings": [
    {
      "frame_idx": 0,
      "team_a_name": "TAPIA / CHINGOTTO",
      "team_b_name": "LEBRON / GALAN",
      "team_a_scores": [0],
      "team_b_scores": [0],
      "confidence": 0.92
    },
    {
      "frame_idx": 300,
      "team_a_name": "TAPIA / CHINGOTTO",
      "team_b_name": "LEBRON / GALAN",
      "team_a_scores": [1],
      "team_b_scores": [0],
      "confidence": 0.89
    }
  ],

  "reconciliation": {
    "fsm_points_confirmed": 5,
    "fsm_points_removed": 1,
    "ocr_points_inserted": 0,
    "corrections": [
      "REMOVED frame 1234: point_won (no OCR confirmation)"
    ]
  }
}
```

Notes:
- `action_predictions` and `action_type` are only populated when action recognition is enabled.
- `rally_states` is always present — rally state classification is a mandatory pipeline step (heuristic at minimum, model when provided).
- `confidence` on low-level events: 1.0 for unambiguous classifications, lower values for events resolved by trajectory analysis in the near-wall overlap zone.
- `ocr_readings` and `reconciliation` are always present — OCR score reading is a mandatory pipeline step.
- When reconciliation removes a FSM point, the cascade events (game_over, set_over, etc.) that followed it are also removed.

## Coordinate Systems

| System | Origin | X | Y | Units |
|--------|--------|---|---|-------|
| **Image** | Top-left of frame | Right | Down | Pixels |
| **Court** | Far-left floor corner | Right (width) | Away from camera (length) | Meters |

Court ranges: X ∈ [0, 10], Y ∈ [0, 20]. Net at Y = 10. "Far" = Y < 10, "Near" = Y >= 10.

## Event Types

### Low-level (`low_level_events[].type`)

| Type | Description | Key fields |
|------|-------------|------------|
| `floor_bounce` | Ball bounced on court floor | `surface`, `court_side` |
| `wall_bounce` | Ball hit a wall | `surface` (wall_far/near/left/right), `court_side` |
| `racket_hit` | Player hit the ball | `player_track_id`, `action_type` (if model enabled) |
| `net_hit` | Ball hit the net | — |

### High-level (`high_level_events[].type`)

| Type | Description |
|------|-------------|
| `serve` | Serve initiated |
| `rally_hit` | Hit during rally |
| `point_won` | Point awarded (with `reason`, `winning_team`). `reason` can be `"ocr_detected"` if inserted by reconciler |
| `double_bounce` | Two bounces on same side |
| `net_fault` | Ball hit net during rally |
| `serve_fault` | Serve fault (with `reason`) |
| `game_over` | Game completed |
| `set_over` | Set completed |
| `match_over` | Match completed |
| `side_switch` | Teams switched sides |

### Score Format

`"Sets: [6-4 | 3-6] Game: 2-1 Points: 30-15"`
