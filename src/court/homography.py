from __future__ import annotations

import cv2
import numpy as np


class HomographyTransformer:
    """Computes and applies perspective transform from image coords to
    real-world court coordinates (bird's-eye view in meters)."""

    def __init__(self, cfg: dict):
        self.court_width: float = cfg["real_width"]
        self.court_length: float = cfg["real_length"]
        self.real_points = np.array(cfg["real_points"], dtype=np.float32)
        self.H: np.ndarray | None = None
        self.H_inv: np.ndarray | None = None

    def compute(self, image_corners: np.ndarray) -> np.ndarray:
        """Compute homography from 4 image corners to real-world court coords.

        Args:
            image_corners: shape (4, 2) in pixel coordinates
                           order: top-left, top-right, bottom-right, bottom-left

        Returns:
            3x3 homography matrix
        """
        if image_corners.shape != (4, 2):
            raise ValueError(f"Expected 4 corners, got shape {image_corners.shape}")

        self.H, status = cv2.findHomography(image_corners, self.real_points)
        if self.H is None:
            raise RuntimeError("Homography computation failed")

        self.H_inv = np.linalg.inv(self.H)
        return self.H

    def image_to_court(self, points: np.ndarray) -> np.ndarray:
        """Transform points from image coordinates to court coordinates.

        Args:
            points: shape (N, 2) or (2,) pixel coordinates

        Returns:
            shape (N, 2) or (2,) court coordinates in meters
        """
        if self.H is None:
            raise RuntimeError("Homography not computed yet. Call compute() first.")

        single = points.ndim == 1
        if single:
            points = points.reshape(1, 2)

        pts = points.astype(np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.H)
        result = transformed.reshape(-1, 2)

        return result[0] if single else result

    def court_to_image(self, points: np.ndarray) -> np.ndarray:
        """Transform points from court coordinates back to image coordinates."""
        if self.H_inv is None:
            raise RuntimeError("Homography not computed yet. Call compute() first.")

        single = points.ndim == 1
        if single:
            points = points.reshape(1, 2)

        pts = points.astype(np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.H_inv)
        result = transformed.reshape(-1, 2)

        return result[0] if single else result

    def is_on_court(self, court_point: np.ndarray, margin: float = 0.5) -> bool:
        """Check if a point (in court coords) is within the court boundaries."""
        x, y = court_point
        return (
            -margin <= x <= self.court_width + margin
            and -margin <= y <= self.court_length + margin
        )

    def get_court_side(self, court_point: np.ndarray) -> str:
        """Determine which side of the court a point is on.

        Returns 'near' (bottom half, closer to camera) or 'far' (top half).
        """
        half = self.court_length / 2
        return "near" if court_point[1] >= half else "far"

    def get_service_box(self, side: str, position: str) -> np.ndarray:
        """Get the service box corners for padel.

        Padel service boxes: each side split into left/right by center line,
        and bounded by the service line (3m from net on each side).

        Args:
            side: 'near' or 'far'
            position: 'left' or 'right'

        Returns:
            (4, 2) array of corner coordinates in meters
        """
        half_w = self.court_width / 2
        net_y = self.court_length / 2
        service_dist = 3.0  # meters from net

        if side == "far":
            y_near = net_y - service_dist
            y_far = 0.0
        else:
            y_near = net_y + service_dist
            y_far = self.court_length

        if position == "left":
            x_left, x_right = 0.0, half_w
        else:
            x_left, x_right = half_w, self.court_width

        return np.array([
            [x_left, y_near], [x_right, y_near],
            [x_right, y_far], [x_left, y_far]
        ], dtype=np.float32)
