from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO


@dataclass
class TrackedPlayer:
    track_id: int
    bbox: np.ndarray           # [x1, y1, x2, y2]
    foot_position: np.ndarray  # [x, y]
    center: np.ndarray         # [x, y]
    confidence: float
    team_id: int = -1          # assigned later
    crop: np.ndarray | None = None


@dataclass
class DetectedRacket:
    bbox: np.ndarray           # [x1, y1, x2, y2]
    center: np.ndarray         # [x, y]
    confidence: float
    owner_track_id: int | None = None  # nearest player's track_id


@dataclass
class FrameTracking:
    frame_idx: int
    players: list[TrackedPlayer] = field(default_factory=list)
    rackets: list[DetectedRacket] = field(default_factory=list)


class PlayerTracker:
    """Tracks players and detects rackets in a single YOLO forward pass.

    Uses BoT-SORT (or ByteTrack) for persistent player IDs.  Racket
    detections come from the same inference and are associated with the
    nearest player — no tracking needed for rackets.
    """

    def __init__(self, det_cfg: dict, track_cfg: dict):
        self.model = YOLO(det_cfg["model"])
        self.person_class_id = det_cfg["person_class_id"]
        self.imgsz = det_cfg["imgsz"]
        self.tracker = track_cfg["tracker"]
        self.persist = track_cfg["persist"]

        racket_cfg = det_cfg.get("racket", {})
        self.racket_class_id = racket_cfg.get("class_id", 38)
        self.racket_detection_enabled = racket_cfg.get("enabled", True)

        self.detect_classes = [self.person_class_id]
        if self.racket_detection_enabled:
            self.detect_classes.append(self.racket_class_id)

        self.confidence = min(
            det_cfg["confidence"],
            racket_cfg.get("confidence", 0.3),
        ) if self.racket_detection_enabled else det_cfg["confidence"]

        self.person_confidence = det_cfg["confidence"]
        self.racket_confidence = racket_cfg.get("confidence", 0.3)

    def track_frame(self, frame: np.ndarray, frame_idx: int) -> FrameTracking:
        """Single YOLO pass for both players and rackets."""
        results = self.model.track(
            frame,
            conf=self.confidence,
            classes=self.detect_classes,
            imgsz=self.imgsz,
            tracker=self.tracker,
            persist=self.persist,
            verbose=False,
        )

        ft = FrameTracking(frame_idx=frame_idx)

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            has_ids = boxes.id is not None

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

                if cls_id == self.person_class_id:
                    if not has_ids:
                        continue
                    if conf < self.person_confidence:
                        continue
                    tid = int(boxes.id[i].cpu().numpy())
                    foot = np.array([(x1 + x2) / 2, y2])
                    crop = frame[int(y1):int(y2), int(x1):int(x2)].copy()
                    ft.players.append(TrackedPlayer(
                        track_id=tid,
                        bbox=xyxy,
                        foot_position=foot,
                        center=center,
                        confidence=conf,
                        crop=crop,
                    ))

                elif cls_id == self.racket_class_id:
                    if conf < self.racket_confidence:
                        continue
                    ft.rackets.append(DetectedRacket(
                        bbox=xyxy,
                        center=center,
                        confidence=conf,
                    ))

        # associate rackets with nearest player after all players are collected
        for racket in ft.rackets:
            racket.owner_track_id = self._find_nearest_player(racket.center, ft)

        return ft

    @staticmethod
    def _find_nearest_player(
        point: np.ndarray, ft: FrameTracking
    ) -> int | None:
        """Return track_id of the player closest to the given point."""
        min_dist = float("inf")
        nearest_id = None
        for player in ft.players:
            dist = float(np.linalg.norm(point - player.center))
            if dist < min_dist:
                min_dist = dist
                nearest_id = player.track_id
        return nearest_id

    def reset(self) -> None:
        """Reset tracker state (for new video)."""
        self.model = YOLO(self.model.model_name)
