from __future__ import annotations

import cv2
import numpy as np


class CourtDetector:
    """Detects padel court corners in the frame.

    Supports two modes:
    - manual: corners are provided via config (4 floor + 4 wall-top)
    - auto: floor corners detected via line detection (wall tops still need manual input)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mode = cfg["mode"]
        self.floor_corners: np.ndarray | None = None
        self.wall_top_corners: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Detect court geometry points.

        Returns:
            dict with keys:
                'floor': (4, 2) array of floor corners
                'wall_tops': (4, 2) array of wall top corners
                'all_points': (8, 2) combined array for CourtGeometry
        """
        if self.mode == "manual":
            return self._manual(frame)
        else:
            return self._auto(frame)

    def _manual(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        floor_pts = self.cfg["manual_points"]["floor"]
        wall_top_pts = self.cfg["manual_points"]["wall_tops"]

        if not floor_pts or len(floor_pts) != 4:
            raise ValueError(
                "Manual mode requires 4 floor corner points in "
                "config.court.manual_points.floor. "
                "Order: far-left, far-right, near-right, near-left"
            )

        if not wall_top_pts or len(wall_top_pts) != 4:
            raise ValueError(
                "Manual mode requires 4 wall top corner points in "
                "config.court.manual_points.wall_tops. "
                "Order: far-left-top, far-right-top, near-right-top, near-left-top"
            )

        floor = np.array(floor_pts, dtype=np.float32)
        wall_tops = np.array(wall_top_pts, dtype=np.float32)
        self._validate_points(floor, frame.shape)
        self._validate_points(wall_tops, frame.shape)

        self.floor_corners = floor
        self.wall_top_corners = wall_tops

        all_points = np.vstack([floor, wall_tops])
        return {
            "floor": floor,
            "wall_tops": wall_tops,
            "all_points": all_points,
        }

    def _auto(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Auto-detect floor corners via line detection.

        Wall top corners must still be provided manually since they
        are harder to detect automatically from line features.
        """
        auto_cfg = self.cfg["auto"]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(
            blurred, auto_cfg["canny_low"], auto_cfg["canny_high"]
        )

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=auto_cfg["hough_threshold"],
            minLineLength=auto_cfg["min_line_length"],
            maxLineGap=auto_cfg["max_line_gap"],
        )

        if lines is None or len(lines) < 4:
            raise RuntimeError(
                "Auto court detection failed: not enough lines found. "
                "Try adjusting canny/hough parameters or use manual mode."
            )

        floor = self._lines_to_corners(lines, frame.shape)
        self.floor_corners = floor

        wall_top_pts = self.cfg["manual_points"].get("wall_tops", [])
        if not wall_top_pts or len(wall_top_pts) != 4:
            raise ValueError(
                "Auto mode detects floor corners but wall top corners "
                "must be provided in config.court.manual_points.wall_tops"
            )

        wall_tops = np.array(wall_top_pts, dtype=np.float32)
        self.wall_top_corners = wall_tops

        all_points = np.vstack([floor, wall_tops])
        return {
            "floor": floor,
            "wall_tops": wall_tops,
            "all_points": all_points,
        }

    def _lines_to_corners(
        self, lines: np.ndarray, frame_shape: tuple
    ) -> np.ndarray:
        h, w = frame_shape[:2]
        angles = []
        line_list = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            angles.append(angle)
            line_list.append((x1, y1, x2, y2))

        angles = np.array(angles)
        horizontal_mask = (angles < 30) | (angles > 150)
        vertical_mask = (angles > 60) & (angles < 120)

        h_lines = [l for l, m in zip(line_list, horizontal_mask) if m]
        v_lines = [l for l, m in zip(line_list, vertical_mask) if m]

        if len(h_lines) < 2 or len(v_lines) < 2:
            raise RuntimeError(
                "Auto court detection: insufficient line groups. "
                "Use manual mode."
            )

        h_lines_sorted = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        v_lines_sorted = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)

        top_line = h_lines_sorted[0]
        bottom_line = h_lines_sorted[-1]
        left_line = v_lines_sorted[0]
        right_line = v_lines_sorted[-1]

        tl = self._line_intersection(top_line, left_line)
        tr = self._line_intersection(top_line, right_line)
        br = self._line_intersection(bottom_line, right_line)
        bl = self._line_intersection(bottom_line, left_line)

        corners = np.array([tl, tr, br, bl], dtype=np.float32)
        corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)

        return corners

    @staticmethod
    def _line_intersection(line1: tuple, line2: tuple) -> tuple[float, float]:
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return ((x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    @staticmethod
    def _validate_points(points: np.ndarray, frame_shape: tuple) -> None:
        h, w = frame_shape[:2]
        if np.any(points < 0) or np.any(points[:, 0] >= w) or np.any(points[:, 1] >= h):
            raise ValueError(
                f"Points out of frame bounds ({w}x{h}): {points.tolist()}"
            )
