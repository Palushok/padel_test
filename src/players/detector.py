from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class PlayerDetection:
    bbox: np.ndarray       # [x1, y1, x2, y2]
    confidence: float
    foot_position: np.ndarray  # [x, y] — bottom center of bbox (ground contact)
    center: np.ndarray     # [x, y] — center of bbox
    crop: np.ndarray | None = None  # cropped image for team assignment

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class PlayerDetector:
    """Detects players (persons) using YOLO."""

    def __init__(self, cfg: dict):
        self.model = YOLO(cfg["model"])
        self.confidence = cfg["confidence"]
        self.person_class_id = cfg["person_class_id"]
        self.imgsz = cfg["imgsz"]

    def detect(self, frame: np.ndarray) -> list[PlayerDetection]:
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[self.person_class_id],
            imgsz=self.imgsz,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                x1, y1, x2, y2 = xyxy
                foot = np.array([(x1 + x2) / 2, y2])
                center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

                crop = frame[int(y1):int(y2), int(x1):int(x2)].copy()

                detections.append(PlayerDetection(
                    bbox=xyxy,
                    confidence=conf,
                    foot_position=foot,
                    center=center,
                    crop=crop,
                ))

        return detections
