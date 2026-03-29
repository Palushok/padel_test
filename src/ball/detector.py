from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class BallDetection:
    position: np.ndarray  # [x, y] center
    confidence: float
    bbox: np.ndarray      # [x1, y1, x2, y2]
    frame_idx: int = -1


# TODO: Replace YOLO with a specialised ball-tracking model (e.g. TrackNet v3)
#       for significantly higher detection rate on small fast-moving balls.
class BallDetector:
    """Detects the ball using YOLO (sports_ball class).

    For better results, consider fine-tuning on padel data
    or switching to TrackNet.
    """

    def __init__(self, cfg: dict):
        self.model = YOLO(cfg["model"])
        self.confidence = cfg["confidence"]
        self.ball_class_id = cfg["sports_ball_class_id"]
        self.imgsz = cfg["imgsz"]

    def detect(self, frame: np.ndarray, frame_idx: int = -1) -> BallDetection | None:
        """Detect ball in a single frame. Returns the highest-confidence
        detection or None if no ball is found."""
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[self.ball_class_id],
            imgsz=self.imgsz,
            verbose=False,
        )

        best: BallDetection | None = None

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                x1, y1, x2, y2 = xyxy
                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

                det = BallDetection(
                    position=center,
                    confidence=conf,
                    bbox=xyxy,
                    frame_idx=frame_idx,
                )

                if best is None or conf > best.confidence:
                    best = det

        return best
