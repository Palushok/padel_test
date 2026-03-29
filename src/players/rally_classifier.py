from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.players.tracker import FrameTracking

logger = logging.getLogger(__name__)


@dataclass
class RallyState:
    """Per-frame rally state classification."""
    frame_idx: int
    is_rally_active: bool
    confidence: float
    source: str  # "heuristic" or "model"


class RallyClassificationModel:
    """MobileNetV3-Small binary classifier for rally state.

    Input: full frame downscaled to input_size.
    Output: P(rally_active) via sigmoid.
    """

    def __init__(self, model_path: str, input_size: tuple[int, int], device: str | None = None):
        import torch
        import torchvision.models as models
        import torchvision.transforms as T

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size

        backbone = models.mobilenet_v3_small(weights=None)
        in_features = backbone.classifier[0].in_features
        backbone.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),
        )
        self.model = backbone

        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, frame: np.ndarray) -> float:
        """Return P(rally_active) for a single frame."""
        import torch

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(tensor).squeeze()
            prob = torch.sigmoid(logit).item()
        return prob


class RallyClassifier:
    """Mandatory rally state classifier with heuristic base and optional model.

    Always produces a rally state for every frame. When a trained model is
    provided, model predictions override the heuristic on sampled frames
    and are interpolated across the rest.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model: RallyClassificationModel | None = None
        self.sample_interval = cfg.get("sample_interval", 5)

        model_path = cfg.get("model_path")
        if model_path:
            input_size = tuple(cfg.get("input_size", [224, 224]))
            device = cfg.get("device")
            self.model = RallyClassificationModel(model_path, input_size, device)
            logger.info(f"Rally classifier model loaded from {model_path}")
        else:
            logger.info("Rally classifier: model not provided, using heuristic only")

        heuristic_cfg = cfg.get("heuristic", {})
        self.speed_threshold = heuristic_cfg.get("speed_threshold", 3.0)
        self.active_ratio = heuristic_cfg.get("active_ratio", 0.5)
        self.window_frames = heuristic_cfg.get("window_frames", 30)

    # ── Phase 2: model-based sampling ──

    def predict_frame(self, frame: np.ndarray, frame_idx: int) -> RallyState | None:
        """Run model on a single frame (called during Phase 2 loop).

        Returns None if model is not loaded or this frame is not sampled.
        """
        if self.model is None:
            return None
        if frame_idx % self.sample_interval != 0:
            return None

        prob = self.model.predict(frame)
        return RallyState(
            frame_idx=frame_idx,
            is_rally_active=prob > 0.5,
            confidence=abs(prob - 0.5) * 2.0,
            source="model",
        )

    # ── Phase 3: heuristic + interpolation ──

    def classify_all(
        self,
        frame_trackings: list[FrameTracking],
        total_frames: int,
        model_predictions: list[RallyState] | None = None,
    ) -> list[RallyState]:
        """Produce a RallyState for every frame.

        Strategy:
        1. Compute heuristic for all frames from player movement speeds.
        2. If model predictions exist, override heuristic on sampled frames
           and interpolate between samples.
        """
        heuristic = self._compute_heuristic(frame_trackings, total_frames)

        if not model_predictions:
            return heuristic

        model_map: dict[int, RallyState] = {s.frame_idx: s for s in model_predictions}
        sorted_frames = sorted(model_map.keys())

        if not sorted_frames:
            return heuristic

        result = list(heuristic)

        for fidx in sorted_frames:
            if fidx < len(result):
                result[fidx] = model_map[fidx]

        for i in range(len(sorted_frames) - 1):
            f_start = sorted_frames[i]
            f_end = sorted_frames[i + 1]
            prob_start = model_map[f_start].confidence if model_map[f_start].is_rally_active else -model_map[f_start].confidence
            prob_end = model_map[f_end].confidence if model_map[f_end].is_rally_active else -model_map[f_end].confidence

            for f in range(f_start + 1, f_end):
                if f >= len(result):
                    break
                t = (f - f_start) / (f_end - f_start)
                interp = prob_start + t * (prob_end - prob_start)
                is_active = interp > 0
                result[f] = RallyState(
                    frame_idx=f,
                    is_rally_active=is_active,
                    confidence=abs(interp),
                    source="model_interpolated",
                )

        return result

    def _compute_heuristic(
        self,
        frame_trackings: list[FrameTracking],
        total_frames: int,
    ) -> list[RallyState]:
        """Heuristic rally detection from player movement speeds."""
        speeds = self._compute_player_speeds(frame_trackings)

        half_w = self.window_frames // 2
        states: list[RallyState] = []

        for i in range(total_frames):
            w_start = max(0, i - half_w)
            w_end = min(total_frames, i + half_w + 1)

            window_speeds = speeds[w_start:w_end]
            if not window_speeds:
                states.append(RallyState(
                    frame_idx=i, is_rally_active=False, confidence=0.0, source="heuristic",
                ))
                continue

            moving_count = sum(1 for s in window_speeds if s > self.speed_threshold)
            ratio = moving_count / len(window_speeds)

            is_active = ratio >= self.active_ratio
            confidence = min(1.0, abs(ratio - self.active_ratio) / self.active_ratio)

            states.append(RallyState(
                frame_idx=i,
                is_rally_active=is_active,
                confidence=confidence,
                source="heuristic",
            ))

        return states

    @staticmethod
    def _compute_player_speeds(
        frame_trackings: list[FrameTracking],
    ) -> list[float]:
        """Average player movement speed per frame (pixels/frame)."""
        speeds: list[float] = []
        prev_positions: dict[int, np.ndarray] = {}

        for ft in frame_trackings:
            frame_speeds: list[float] = []
            current_positions: dict[int, np.ndarray] = {}

            for player in ft.players:
                current_positions[player.track_id] = player.foot_position
                if player.track_id in prev_positions:
                    delta = np.linalg.norm(
                        player.foot_position - prev_positions[player.track_id]
                    )
                    frame_speeds.append(float(delta))

            avg_speed = float(np.mean(frame_speeds)) if frame_speeds else 0.0
            speeds.append(avg_speed)
            prev_positions = current_positions

        return speeds
