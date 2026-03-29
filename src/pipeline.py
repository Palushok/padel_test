from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.ball.detector import BallDetector
from src.ball.interpolator import BallInterpolator
from src.ball.kalman_filter import BallKalmanFilter
from src.ball.trajectory_analyzer import TrajectoryAnalyzer
from src.court.detector import CourtDetector
from src.court.geometry import CourtGeometry
from src.court.homography import HomographyTransformer
from src.events.high_level import GameStateMachine
from src.events.low_level import LowLevelEventDetector
from src.players.rally_classifier import RallyClassifier
from src.players.team_assigner import TeamAssigner
from src.players.tracker import PlayerTracker
from src.score.ocr import ScoreOCR
from src.score.reconciler import ScoreReconciler
from src.visualization.renderer import Renderer

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    total_frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    frame_trackings: list = field(default_factory=list)
    ball_detections: list = field(default_factory=list)
    ball_states: list = field(default_factory=list)
    ball_interpolated: list = field(default_factory=list)
    team_assignments: dict = field(default_factory=dict)
    action_predictions: list = field(default_factory=list)
    rally_states: list = field(default_factory=list)
    low_level_events: list = field(default_factory=list)
    high_level_events: list = field(default_factory=list)
    ocr_readings: list = field(default_factory=list)
    reconciliation: Any = None
    court_corners: np.ndarray | None = None
    court_all_points: np.ndarray | None = None
    score: Any = None


class Pipeline:
    """Main orchestrator for the padel analysis pipeline.

    Phases:
    1. Initialization: court detection (8 points), homography, 3D geometry
    2. Frame processing: detect/track players, detect ball
    3. Post-processing: filter ball, interpolate, (optional) action recognition,
       detect events, run game FSM, OCR score reading + reconciliation
    4. Visualization: render output video with overlays + minimap
    """

    def __init__(self, config_path: str, video_path: str | None = None):
        self.cfg = self._load_config(config_path)
        if video_path:
            self.cfg["video"]["path"] = video_path
        self._init_components()

    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _init_components(self) -> None:
        self.court_detector = CourtDetector(self.cfg["court"])
        self.homography = HomographyTransformer(self.cfg["court"])
        self.court_geometry = CourtGeometry()

        self.player_tracker = PlayerTracker(
            self.cfg["players"]["detection"],
            self.cfg["players"]["tracking"],
        )
        self.team_assigner = TeamAssigner()

        ar_cfg = self.cfg["players"].get("action_recognition", {})
        self.action_recognition_enabled = ar_cfg.get("enabled", False)
        self.action_recognizer = None
        if self.action_recognition_enabled:
            from src.players.action_recognition import ActionRecognizer
            self.action_recognizer = ActionRecognizer(ar_cfg)

        self.ball_detector = BallDetector(self.cfg["ball"]["detection"])
        self.ball_filter = BallKalmanFilter(self.cfg["ball"]["kalman"])
        self.ball_interpolator = BallInterpolator(self.cfg["ball"]["interpolation"])

        self.rally_classifier = RallyClassifier(
            self.cfg.get("rally_classifier", {})
        )

        self._trajectory_analysis_cfg = self.cfg.get("trajectory_analysis", {})

        ocr_cfg = self.cfg.get("score_ocr", {})
        self.score_ocr = ScoreOCR(ocr_cfg)
        self._ocr_reconciliation_cfg = ocr_cfg.get("reconciliation", {})
        self.score_reconciler: ScoreReconciler | None = None

    def run(self) -> PipelineResult:
        video_path = self.cfg["video"]["path"]
        if not video_path:
            raise ValueError("No video path specified")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video: {video_path}")
        logger.info(f"  Frames: {total_frames}, FPS: {fps}, Size: {width}x{height}")

        result = PipelineResult(
            total_frames=total_frames, fps=fps, width=width, height=height
        )

        # ── Phase 1: Court detection, homography, 3D geometry ──
        logger.info("Phase 1: Court detection & geometry")
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame")

        court_data = self.court_detector.detect(first_frame)
        floor_corners = court_data["floor"]
        all_points = court_data["all_points"]

        self.homography.compute(floor_corners)
        self.court_geometry.build(all_points)

        result.court_corners = floor_corners
        result.court_all_points = all_points
        logger.info(f"  Floor corners: {floor_corners.tolist()}")
        logger.info(f"  Wall tops: {court_data['wall_tops'].tolist()}")
        logger.info("  5 surfaces built: floor + 4 walls")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # ── Phase 2: Frame-by-frame processing ──
        # Single video pass: player tracking, ball detection, and OCR sampling.
        logger.info("Phase 2: Frame-by-frame detection & tracking (+ OCR, rally)")
        frame_trackings = []
        ball_detections = []
        ocr_readings = []
        rally_model_predictions = []
        ocr_interval = self.score_ocr.sample_interval

        t0 = time.time()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ft = self.player_tracker.track_frame(frame, frame_idx)
            frame_trackings.append(ft)

            ball_det = self.ball_detector.detect(frame, frame_idx)
            ball_detections.append(ball_det)

            if frame_idx % ocr_interval == 0:
                reading = self.score_ocr.read_frame(frame, frame_idx)
                if reading.is_valid:
                    ocr_readings.append(reading)

            rally_pred = self.rally_classifier.predict_frame(frame, frame_idx)
            if rally_pred is not None:
                rally_model_predictions.append(rally_pred)

            if frame_idx % 100 == 0:
                elapsed = time.time() - t0
                speed = (frame_idx + 1) / max(elapsed, 0.01)
                logger.info(
                    f"  Frame {frame_idx}/{total_frames} "
                    f"({speed:.1f} fps) "
                    f"Players: {len(ft.players)}, "
                    f"Ball: {'yes' if ball_det else 'no'}"
                )

            frame_idx += 1

        cap.release()
        elapsed_total = time.time() - t0
        logger.info(f"  Processed {frame_idx} frames in {elapsed_total:.1f}s")
        logger.info(f"  OCR: {len(ocr_readings)} valid readings collected")

        result.frame_trackings = frame_trackings
        result.ball_detections = ball_detections
        result.ocr_readings = ocr_readings

        # ── Phase 3: Post-processing ──
        logger.info("Phase 3: Post-processing")

        # 3a. Team assignment (position-based)
        logger.info("  Assigning teams by court position...")
        team_assignments = self.team_assigner.assign_teams(
            frame_trackings, self.homography
        )
        result.team_assignments = team_assignments
        logger.info(f"  Teams: {team_assignments}")

        for ft in frame_trackings:
            for player in ft.players:
                player.team_id = team_assignments.get(player.track_id, -1)

        # 3b. Ball interpolation (first — fills gaps using future data, offline advantage)
        logger.info("  Interpolating missing ball positions...")
        ball_interpolated = self.ball_interpolator.interpolate(
            ball_detections, total_frames
        )
        result.ball_interpolated = ball_interpolated

        # 3c. Ball Kalman filter (on interpolated + detected positions)
        logger.info("  Filtering ball trajectory (Kalman)...")
        det_map = {d.frame_idx: d.position for d in ball_detections if d is not None}
        positions: list[np.ndarray | None] = []
        is_detected: list[bool] = []
        for i in range(total_frames):
            if i in det_map:
                positions.append(det_map[i])
                is_detected.append(True)
            elif ball_interpolated[i] is not None:
                positions.append(ball_interpolated[i])
                is_detected.append(False)
            else:
                positions.append(None)
                is_detected.append(False)

        ball_states = self.ball_filter.filter_trajectory(positions, is_detected)
        result.ball_states = ball_states

        # 3d. Action recognition (optional)
        action_predictions = []
        if self.action_recognition_enabled and self.action_recognizer is not None:
            logger.info("  Running action recognition...")
            action_predictions = self.action_recognizer.predict_actions(
                frame_trackings
            )
            result.action_predictions = action_predictions
            logger.info(f"  Found {len(action_predictions)} action predictions")
        else:
            logger.info("  Action recognition disabled — using rule-based hit detection only")

        # 3e. Rally state classification (heuristic + optional model)
        logger.info("  Classifying rally state...")
        rally_states = self.rally_classifier.classify_all(
            frame_trackings, total_frames,
            rally_model_predictions or None,
        )
        result.rally_states = rally_states
        active_count = sum(1 for rs in rally_states if rs.is_rally_active)
        logger.info(f"  Rally active: {active_count}/{len(rally_states)} frames")

        # 3f. Trajectory analyzer (for ambiguous surface classification)
        trajectory_analyzer = None
        if self._trajectory_analysis_cfg.get("enabled", True):
            trajectory_analyzer = TrajectoryAnalyzer(
                self._trajectory_analysis_cfg, self.homography,
            )
            logger.info("  Trajectory analyzer enabled")

        # 3g. Low-level events (using 3D geometry + trajectory analysis)
        logger.info("  Detecting low-level events...")
        ll_detector = LowLevelEventDetector(
            self.cfg["events"]["low_level"],
            self.homography,
            self.court_geometry,
            trajectory_analyzer=trajectory_analyzer,
            rally_states=rally_states,
        )
        low_level_events = ll_detector.detect(
            ball_states, frame_trackings, action_predictions or None
        )
        result.low_level_events = low_level_events
        logger.info(f"  Found {len(low_level_events)} low-level events")

        for evt in low_level_events:
            logger.debug(
                f"    Frame {evt.frame_idx}: {evt.event_type.value}"
                f" surface={evt.surface} action={evt.action_type}"
            )

        # 3h. High-level game FSM
        logger.info("  Processing high-level game events...")
        game_fsm = GameStateMachine(self.cfg["events"]["high_level"])
        high_level_events = game_fsm.process(low_level_events, team_assignments)
        result.high_level_events = high_level_events
        result.score = game_fsm.score
        logger.info(f"  Found {len(high_level_events)} high-level events")
        logger.info(f"  Final score: {game_fsm.score.to_string()}")

        # 3i. OCR reconciliation (readings already collected in Phase 2)
        logger.info("  Reconciling FSM events with OCR score...")
        self.score_reconciler = ScoreReconciler(
            self._ocr_reconciliation_cfg, fps=fps
        )
        reconciliation = self.score_reconciler.reconcile(
            high_level_events, ocr_readings
        )
        result.reconciliation = reconciliation
        result.high_level_events = reconciliation.corrected_events

        logger.info(
            f"  Reconciliation: "
            f"{reconciliation.fsm_points_confirmed} confirmed, "
            f"{reconciliation.fsm_points_removed} removed, "
            f"{reconciliation.ocr_points_inserted} inserted"
        )

        return result

    def render_output(self, result: PipelineResult) -> str:
        video_path = self.cfg["video"]["path"]
        vis_cfg = self.cfg["visualization"]
        output_path = vis_cfg["output_path"]

        cap = cv2.VideoCapture(video_path)
        fps = vis_cfg.get("output_fps") or result.fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path, fourcc, fps, (result.width, result.height)
        )

        renderer = Renderer(vis_cfg, self.homography)

        ll_event_map: dict[int, list] = {}
        for evt in result.low_level_events:
            ll_event_map.setdefault(evt.frame_idx, []).append(evt)

        hl_event_map: dict[int, list] = {}
        for evt in result.high_level_events:
            hl_event_map.setdefault(evt.frame_idx, []).append(evt)

        ball_trail: list[np.ndarray | None] = []

        logger.info(f"Rendering output to {output_path}...")

        frame_idx = 0
        score_text = ""

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tracking = (
                result.frame_trackings[frame_idx]
                if frame_idx < len(result.frame_trackings) else None
            )

            ball_pos = None
            if frame_idx < len(result.ball_states):
                bs = result.ball_states[frame_idx]
                if bs.is_observed or (
                    frame_idx < len(result.ball_interpolated)
                    and result.ball_interpolated[frame_idx] is not None
                ):
                    ball_pos = bs.position

            ball_trail.append(ball_pos)

            for hl_evt in hl_event_map.get(frame_idx, []):
                if "score" in hl_evt.details:
                    score_text = hl_evt.details["score"]
                elif hl_evt.details.get("score_before"):
                    score_text = hl_evt.details["score_before"]

            nearby_events = []
            for offset in range(-15, 16):
                fidx = frame_idx + offset
                if fidx in ll_event_map:
                    nearby_events.extend(ll_event_map[fidx])

            out_frame = renderer.render_frame(
                frame=frame,
                frame_idx=frame_idx,
                court_corners=result.court_corners,
                tracking=tracking,
                ball_position=ball_pos,
                ball_trail=ball_trail,
                events=nearby_events,
                team_assignments=result.team_assignments,
                score_text=score_text,
            )

            writer.write(out_frame)

            if frame_idx % 200 == 0:
                logger.info(f"  Rendered {frame_idx}/{result.total_frames}")

            frame_idx += 1

        cap.release()
        writer.release()
        logger.info(f"Output saved to {output_path}")
        return output_path

    @staticmethod
    def export_results(result: PipelineResult, output_path: str) -> None:
        data: dict[str, Any] = {
            "metadata": {
                "total_frames": result.total_frames,
                "fps": result.fps,
                "width": result.width,
                "height": result.height,
            },
            "team_assignments": {
                str(k): v for k, v in result.team_assignments.items()
            },
            "player_tracking": [],
            "ball_tracking": [],
            "action_predictions": [],
            "low_level_events": [],
            "high_level_events": [],
        }

        for ft in result.frame_trackings:
            frame_data = {
                "frame_idx": ft.frame_idx,
                "players": [
                    {
                        "track_id": p.track_id,
                        "bbox": p.bbox.tolist(),
                        "foot_position": p.foot_position.tolist(),
                        "team_id": p.team_id,
                        "confidence": p.confidence,
                    }
                    for p in ft.players
                ],
                "rackets": [
                    {
                        "bbox": r.bbox.tolist(),
                        "center": r.center.tolist(),
                        "confidence": r.confidence,
                        "owner_track_id": r.owner_track_id,
                    }
                    for r in ft.rackets
                ],
            }
            data["player_tracking"].append(frame_data)

        for bs in result.ball_states:
            data["ball_tracking"].append({
                "frame_idx": bs.frame_idx,
                "position": bs.position.tolist(),
                "velocity": bs.velocity.tolist(),
                "is_observed": bs.is_observed,
            })

        for ap in result.action_predictions:
            data["action_predictions"].append({
                "track_id": ap.track_id,
                "action": ap.action.value,
                "confidence": ap.confidence,
                "frame_start": ap.frame_start,
                "frame_end": ap.frame_end,
                "probabilities": ap.probabilities,
            })

        for evt in result.low_level_events:
            data["low_level_events"].append({
                "type": evt.event_type.value,
                "frame_idx": evt.frame_idx,
                "ball_position_image": evt.ball_position_image.tolist(),
                "ball_position_court": (
                    evt.ball_position_court.tolist()
                    if evt.ball_position_court is not None else None
                ),
                "player_track_id": evt.player_track_id,
                "court_side": evt.court_side,
                "surface": evt.surface,
                "action_type": evt.action_type,
                "confidence": evt.confidence,
            })

        for evt in result.high_level_events:
            data["high_level_events"].append({
                "type": evt.event_type.value,
                "frame_idx": evt.frame_idx,
                "details": evt.details,
            })

        if result.rally_states:
            data["rally_states"] = [
                {
                    "frame_idx": rs.frame_idx,
                    "is_rally_active": rs.is_rally_active,
                    "confidence": rs.confidence,
                    "source": rs.source,
                }
                for rs in result.rally_states
            ]

        if result.ocr_readings:
            data["ocr_readings"] = [
                {
                    "frame_idx": r.frame_idx,
                    "team_a_name": r.team_a_name,
                    "team_b_name": r.team_b_name,
                    "team_a_scores": r.team_a_scores,
                    "team_b_scores": r.team_b_scores,
                    "confidence": r.confidence,
                }
                for r in result.ocr_readings
            ]

        if result.reconciliation:
            data["reconciliation"] = {
                "fsm_points_confirmed": result.reconciliation.fsm_points_confirmed,
                "fsm_points_removed": result.reconciliation.fsm_points_removed,
                "ocr_points_inserted": result.reconciliation.ocr_points_inserted,
                "corrections": result.reconciliation.corrections_log,
            }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results exported to {output_path}")
