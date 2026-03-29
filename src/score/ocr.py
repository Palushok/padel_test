from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScoreReading:
    """A single OCR score reading from one frame."""
    frame_idx: int
    raw_text: str
    team_a_name: str | None = None
    team_b_name: str | None = None
    # Scores can be multi-level: sets, games, points
    # We store the raw numbers per team as parsed from the overlay
    team_a_scores: list[int] = field(default_factory=list)
    team_b_scores: list[int] = field(default_factory=list)
    confidence: float = 0.0
    is_valid: bool = False

    @property
    def team_a_games(self) -> int:
        """Last score column = current games (most common broadcast format)."""
        return self.team_a_scores[-1] if self.team_a_scores else 0

    @property
    def team_b_games(self) -> int:
        return self.team_b_scores[-1] if self.team_b_scores else 0

    def total_points_diff(self, other: ScoreReading) -> tuple[int, int]:
        """Compare two readings, return (delta_a, delta_b) in rightmost score column."""
        da = self.team_a_games - other.team_a_games
        db = self.team_b_games - other.team_b_games
        return (da, db)


class ScoreOCR:
    """Reads the on-screen score overlay using EasyOCR.

    The score overlay typically has two rows:
      Row 1: [flag] TEAM_A_NAME  score1  score2 ...
      Row 2: [flag] TEAM_B_NAME  score1  score2 ...

    The OCR reads the ROI, extracts text, and parses team names + numbers.
    """

    def __init__(self, cfg: dict):
        import easyocr
        self.reader = easyocr.Reader(
            cfg.get("languages", ["en"]),
            gpu=cfg.get("gpu", True),
            verbose=False,
        )
        self.roi = cfg.get("roi")  # [x1, y1, x2, y2] or None (auto-detect)
        self.sample_interval = cfg.get("sample_interval", 30)
        self.min_confidence = cfg.get("min_confidence", 0.5)
        self._last_valid: ScoreReading | None = None

    def read_scores_from_video(
        self, video_path: str, total_frames: int
    ) -> list[ScoreReading]:
        """Read score overlay from video at regular intervals.

        Args:
            video_path: path to video file
            total_frames: total frame count

        Returns:
            list of ScoreReading at sampled frames (only valid ones)
        """
        cap = cv2.VideoCapture(video_path)
        readings: list[ScoreReading] = []

        for frame_idx in range(0, total_frames, self.sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            reading = self.read_frame(frame, frame_idx)
            if reading.is_valid:
                readings.append(reading)
                self._last_valid = reading

            if frame_idx % (self.sample_interval * 10) == 0:
                logger.info(
                    f"  OCR frame {frame_idx}/{total_frames}, "
                    f"valid readings: {len(readings)}"
                )

        cap.release()
        logger.info(f"  OCR: {len(readings)} valid readings from {total_frames} frames")
        return readings

    def read_frame(self, frame: np.ndarray, frame_idx: int = -1) -> ScoreReading:
        """Read score from a single frame."""
        roi_img = self._extract_roi(frame)
        if roi_img is None or roi_img.size == 0:
            return ScoreReading(frame_idx=frame_idx, raw_text="", is_valid=False)

        results = self.reader.readtext(roi_img)

        if not results:
            return ScoreReading(frame_idx=frame_idx, raw_text="", is_valid=False)

        return self._parse_results(results, frame_idx)

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray | None:
        """Extract the score overlay region from the frame."""
        if self.roi:
            x1, y1, x2, y2 = self.roi
            return frame[y1:y2, x1:x2]

        # Auto-detect: score is typically in the bottom-left or top-left
        # Use a heuristic region
        h, w = frame.shape[:2]
        # Try bottom-left quadrant (common placement)
        return frame[int(h * 0.85):h, 0:int(w * 0.45)]

    def _parse_results(
        self, results: list, frame_idx: int
    ) -> ScoreReading:
        """Parse EasyOCR results into structured score data.

        EasyOCR returns: list of (bbox, text, confidence)

        Strategy:
        1. Sort text blocks top-to-bottom (by y-coordinate)
        2. Group into rows (team A = top row, team B = bottom row)
        3. Extract team name (text) and scores (numbers) from each row
        """
        # Sort by vertical position (top of bounding box)
        sorted_results = sorted(results, key=lambda r: min(p[1] for p in r[0]))

        all_text = " | ".join(r[1] for r in sorted_results)
        avg_conf = np.mean([r[2] for r in sorted_results]) if sorted_results else 0.0

        # Group into two rows by y-coordinate clustering
        rows = self._cluster_into_rows(sorted_results)

        if len(rows) < 2:
            return ScoreReading(
                frame_idx=frame_idx,
                raw_text=all_text,
                confidence=float(avg_conf),
                is_valid=False,
            )

        team_a_row = rows[0]
        team_b_row = rows[1]

        team_a_name, team_a_scores = self._parse_row(team_a_row)
        team_b_name, team_b_scores = self._parse_row(team_b_row)

        is_valid = (
            avg_conf >= self.min_confidence
            and len(team_a_scores) > 0
            and len(team_b_scores) > 0
        )

        return ScoreReading(
            frame_idx=frame_idx,
            raw_text=all_text,
            team_a_name=team_a_name,
            team_b_name=team_b_name,
            team_a_scores=team_a_scores,
            team_b_scores=team_b_scores,
            confidence=float(avg_conf),
            is_valid=is_valid,
        )

    def _cluster_into_rows(
        self, results: list, y_threshold: float = 20.0
    ) -> list[list]:
        """Group OCR results into horizontal rows by y-coordinate."""
        if not results:
            return []

        rows: list[list] = [[results[0]]]

        for result in results[1:]:
            y_center = np.mean([p[1] for p in result[0]])
            prev_y = np.mean([p[1] for p in rows[-1][-1][0]])

            if abs(y_center - prev_y) < y_threshold:
                rows[-1].append(result)
            else:
                rows.append([result])

        return rows

    def _parse_row(
        self, row_results: list
    ) -> tuple[str | None, list[int]]:
        """Extract team name and score numbers from a row of OCR results.

        Each row typically contains: "TEAM_NAME  score1  score2 ..."
        """
        texts = []
        for _, text, _ in row_results:
            texts.append(text.strip())

        full_text = " ".join(texts)

        # Extract all numbers from the row
        numbers = re.findall(r'\b(\d+)\b', full_text)
        scores = [int(n) for n in numbers]

        # Everything that's not a number is the team name
        name_parts = re.sub(r'\b\d+\b', '', full_text).strip()
        name_parts = re.sub(r'\s+', ' ', name_parts).strip()
        # Clean up separators
        name = name_parts.strip("| -/").strip() if name_parts else None

        return name, scores
