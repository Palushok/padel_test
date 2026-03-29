from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


class ActionType(Enum):
    SERVE = "serve"
    RETURN = "return"
    VOLLEY = "volley"
    GROUNDSTROKE = "groundstroke"
    LOB = "lob"
    OVERHEAD = "overhead"
    SMASH = "smash"
    WALL_PLAY = "wall_play"


ACTION_CLASSES = list(ActionType)


@dataclass
class ActionPrediction:
    track_id: int
    action: ActionType
    confidence: float
    frame_start: int
    frame_end: int
    probabilities: dict[str, float]


class ActionRecognitionModel(nn.Module):
    """Video-based action recognition using R3D-18 backbone.

    Input: sequence of player crop frames (B, C, T, H, W)
    Output: action class logits (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int = len(ACTION_CLASSES),
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        from torchvision.models.video import r3d_18, R3D_18_Weights
        weights = R3D_18_Weights.DEFAULT if pretrained_backbone else None
        backbone = r3d_18(weights=weights)

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clips: (B, C, T, H, W) video clips

        Returns:
            (B, num_classes) logits
        """
        features = self.encoder(clips).squeeze(-1).squeeze(-1).squeeze(-1)
        return self.classifier(features)


class ActionRecognizer:
    """Runs action recognition on tracked player crop sequences.

    Sliding window over each player's track: collects crops, resizes,
    feeds through the video classification model.
    """

    def __init__(self, cfg: dict):
        self.clip_length = cfg.get("clip_length", 16)
        self.stride = cfg.get("stride", 8)
        self.crop_size = tuple(cfg.get("crop_size", [112, 112]))
        self.device = cfg.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_path = cfg.get("model_path")

        self.model = ActionRecognitionModel(
            num_classes=len(ACTION_CLASSES),
            pretrained_backbone=True,
        )

        if self.model_path and Path(self.model_path).exists():
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989],
            ),
        ])

    def predict_actions(
        self, frame_trackings: list
    ) -> list[ActionPrediction]:
        """Run action recognition on all tracked players.

        Args:
            frame_trackings: list of FrameTracking per frame

        Returns:
            list of ActionPrediction
        """
        track_data = self._collect_track_data(frame_trackings)
        predictions: list[ActionPrediction] = []

        for track_id, data in track_data.items():
            track_preds = self._predict_track(track_id, data)
            predictions.extend(track_preds)

        return predictions

    def _collect_track_data(
        self, frame_trackings: list
    ) -> dict[int, list[dict]]:
        track_data: dict[int, list[dict]] = {}
        for ft in frame_trackings:
            for player in ft.players:
                entry = {
                    "frame_idx": ft.frame_idx,
                    "bbox": player.bbox,
                    "crop": player.crop,
                }
                track_data.setdefault(player.track_id, []).append(entry)
        return track_data

    def _predict_track(
        self, track_id: int, data: list[dict]
    ) -> list[ActionPrediction]:
        predictions = []
        data_sorted = sorted(data, key=lambda d: d["frame_idx"])

        for start in range(0, len(data_sorted) - self.clip_length + 1, self.stride):
            window = data_sorted[start: start + self.clip_length]
            pred = self._classify_window(track_id, window)
            if pred is not None:
                predictions.append(pred)

        return predictions

    def _classify_window(
        self, track_id: int, window: list[dict]
    ) -> ActionPrediction | None:
        clip_frames = []

        for entry in window:
            crop = entry["crop"]
            if crop is None or crop.size == 0:
                crop = np.zeros((*self.crop_size, 3), dtype=np.uint8)
            frame_tensor = self.transform(crop)
            clip_frames.append(frame_tensor)

        # (C, T, H, W)
        clip_tensor = torch.stack(clip_frames, dim=1).unsqueeze(0)
        clip_tensor = clip_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(clip_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        pred_idx = int(probs.argmax())
        action = ACTION_CLASSES[pred_idx]
        conf = float(probs[pred_idx])

        prob_dict = {
            ac.value: float(probs[i]) for i, ac in enumerate(ACTION_CLASSES)
        }

        return ActionPrediction(
            track_id=track_id,
            action=action,
            confidence=conf,
            frame_start=window[0]["frame_idx"],
            frame_end=window[-1]["frame_idx"],
            probabilities=prob_dict,
        )
