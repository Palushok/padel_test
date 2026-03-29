from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class Surface(Enum):
    FLOOR = "floor"
    WALL_FAR = "wall_far"      # back wall, far from camera
    WALL_NEAR = "wall_near"    # back wall, near to camera
    WALL_LEFT = "wall_left"    # left side wall
    WALL_RIGHT = "wall_right"  # right side wall
    OUTSIDE = "outside"        # not on any surface


# Standard padel court 3D dimensions (meters)
COURT_LENGTH = 20.0
COURT_WIDTH = 10.0
WALL_HEIGHT = 4.0  # total enclosure height (glass top ~3m + solid bottom ~1m)
GLASS_BOTTOM = 1.0  # solid wall height (glass starts above this)


@dataclass
class CourtSurface:
    """A planar surface of the court defined by a polygon in image space."""
    surface_type: Surface
    image_polygon: np.ndarray  # (N, 2) polygon vertices in image coords
    real_3d_corners: np.ndarray  # (N, 3) corresponding 3D corners


class CourtGeometry:
    """Full 3D court geometry built from 8 marked image points.

    The user marks 8 points on the first frame:
      Points 0-3: Floor corners (top-left, top-right, bottom-right, bottom-left
                   as seen in image, i.e. far-left, far-right, near-right, near-left)
      Points 4-7: Wall top-edge corners (same order:
                   far-left-top, far-right-top, near-right-top, near-left-top)

    This defines 5 surfaces:
      - Floor:      points [0,1,2,3]
      - Far wall:   points [4,5,1,0]   (top-left, top-right, floor-right, floor-left)
      - Right wall: points [5,6,2,1]   (far-top, near-top, floor-near, floor-far)
      - Near wall:  points [6,7,3,2]   (top-right, top-left, floor-left, floor-right)
      - Left wall:  points [7,4,0,3]   (near-top, far-top, floor-far, floor-near)
    """

    # 3D coordinates for the 8 court corners (meters)
    # Origin at far-left floor corner, Y along length, Z up
    REAL_3D_POINTS = np.array([
        # Floor corners (z=0)
        [0.0, 0.0, 0.0],             # 0: far-left
        [COURT_WIDTH, 0.0, 0.0],     # 1: far-right
        [COURT_WIDTH, COURT_LENGTH, 0.0],  # 2: near-right
        [0.0, COURT_LENGTH, 0.0],    # 3: near-left
        # Wall top corners (z=WALL_HEIGHT)
        [0.0, 0.0, WALL_HEIGHT],             # 4: far-left-top
        [COURT_WIDTH, 0.0, WALL_HEIGHT],     # 5: far-right-top
        [COURT_WIDTH, COURT_LENGTH, WALL_HEIGHT],  # 6: near-right-top
        [0.0, COURT_LENGTH, WALL_HEIGHT],    # 7: near-left-top
    ], dtype=np.float32)

    SURFACE_DEFINITIONS = {
        Surface.FLOOR:      ([0, 1, 2, 3], "Floor"),
        Surface.WALL_FAR:   ([4, 5, 1, 0], "Far back wall"),
        Surface.WALL_RIGHT: ([5, 6, 2, 1], "Right side wall"),
        Surface.WALL_NEAR:  ([6, 7, 3, 2], "Near back wall"),
        Surface.WALL_LEFT:  ([7, 4, 0, 3], "Left side wall"),
    }

    def __init__(self):
        self.image_points: np.ndarray | None = None  # (8, 2)
        self.surfaces: dict[Surface, CourtSurface] = {}
        self._is_built = False

    def build(self, image_points: np.ndarray) -> None:
        """Build court geometry from 8 marked image points.

        Args:
            image_points: shape (8, 2), order as described in class docstring
        """
        if image_points.shape != (8, 2):
            raise ValueError(
                f"Expected 8 points with shape (8,2), got {image_points.shape}. "
                "Order: floor [far-left, far-right, near-right, near-left], "
                "wall-tops [far-left, far-right, near-right, near-left]"
            )

        self.image_points = image_points.astype(np.float32)
        self.surfaces = {}

        for surface_type, (indices, name) in self.SURFACE_DEFINITIONS.items():
            img_poly = self.image_points[indices]
            real_3d = self.REAL_3D_POINTS[indices]
            self.surfaces[surface_type] = CourtSurface(
                surface_type=surface_type,
                image_polygon=img_poly,
                real_3d_corners=real_3d,
            )

        self._is_built = True

    def classify_point(self, image_point: np.ndarray) -> Surface:
        """Determine which surface an image point lies on.

        Tests point-in-polygon for each surface, prioritizing floor.
        NOTE: This is the legacy image-space method. Prefer
        ``classify_by_court_coords()`` which handles polygon overlap correctly.
        """
        if not self._is_built:
            raise RuntimeError("Court geometry not built. Call build() first.")

        pt = image_point.astype(np.float32)

        if self._point_in_polygon(pt, self.surfaces[Surface.FLOOR].image_polygon):
            return Surface.FLOOR

        for surface_type in [
            Surface.WALL_FAR, Surface.WALL_NEAR,
            Surface.WALL_LEFT, Surface.WALL_RIGHT,
        ]:
            poly = self.surfaces[surface_type].image_polygon
            if self._point_in_polygon(pt, poly):
                return surface_type

        return Surface.OUTSIDE

    def classify_by_court_coords(
        self, court_point: np.ndarray, image_point: np.ndarray | None = None,
    ) -> Surface:
        """Determine which surface the ball bounced off using court coordinates.

        The floor homography maps image points to the floor plane (Z=0).
        If the ball is actually ON the floor, the coordinates fall within
        [0, COURT_WIDTH] x [0, COURT_LENGTH].  If the ball is on a WALL
        (Z > 0), the homography projects it to a position OUTSIDE the court
        boundaries — the direction of the overflow tells us which wall.

        Falls back to image-space polygon test when the ball is near the
        boundary (ambiguous zone).

        Args:
            court_point: ball position in court coords (from floor homography)
            image_point: ball position in image coords (for polygon fallback)

        Returns:
            The surface the ball most likely bounced off.
        """
        if not self._is_built:
            raise RuntimeError("Court geometry not built. Call build() first.")

        cx, cy = float(court_point[0]), float(court_point[1])

        # Well inside the court floor → definitely a floor bounce
        floor_margin = 0.5  # meters — inside this margin, it's floor
        if (floor_margin <= cx <= COURT_WIDTH - floor_margin
                and floor_margin <= cy <= COURT_LENGTH - floor_margin):
            return Surface.FLOOR

        # Clearly outside court bounds → on a wall
        # Determine which wall by which boundary was exceeded most
        overflows = {
            Surface.WALL_FAR:   -cy,                   # cy < 0 → far wall
            Surface.WALL_NEAR:  cy - COURT_LENGTH,     # cy > 20 → near wall
            Surface.WALL_LEFT:  -cx,                   # cx < 0 → left wall
            Surface.WALL_RIGHT: cx - COURT_WIDTH,      # cx > 10 → right wall
        }

        max_wall = max(overflows, key=overflows.get)
        max_overflow = overflows[max_wall]

        if max_overflow > 0:
            return max_wall

        # Ambiguous zone: within [0, margin] or [court_dim - margin, court_dim]
        # near a boundary but still "inside". Use image polygon as tiebreaker.
        if image_point is not None:
            pt = image_point.astype(np.float32)

            # Check walls first (since we're near the edge, wall is more likely
            # than floor in the overlap zone)
            for surface_type in [
                Surface.WALL_NEAR, Surface.WALL_LEFT,
                Surface.WALL_RIGHT, Surface.WALL_FAR,
            ]:
                poly = self.surfaces[surface_type].image_polygon
                if self._point_in_polygon(pt, poly):
                    # Confirm that the overflow direction is consistent
                    if overflows[surface_type] > -floor_margin:
                        return surface_type

        return Surface.FLOOR

    def distance_to_surface(
        self, image_point: np.ndarray, surface_type: Surface
    ) -> float:
        """Distance from image point to the nearest edge of a surface polygon."""
        if surface_type not in self.surfaces:
            return float("inf")
        poly = self.surfaces[surface_type].image_polygon.astype(np.float32)
        return abs(cv2.pointPolygonTest(
            poly.reshape(-1, 1, 2), tuple(image_point.astype(float)), True
        ))

    def nearest_surface(self, image_point: np.ndarray) -> tuple[Surface, float]:
        """Find the nearest surface to an image point and return (surface, distance).

        Distance is negative if inside the polygon (closer = more negative).
        """
        exact = self.classify_point(image_point)
        if exact != Surface.OUTSIDE:
            return exact, 0.0

        best_surface = Surface.OUTSIDE
        best_dist = float("inf")
        pt = tuple(image_point.astype(float))

        for surface_type, cs in self.surfaces.items():
            poly = cs.image_polygon.astype(np.float32).reshape(-1, 1, 2)
            dist = abs(cv2.pointPolygonTest(poly, pt, True))
            if dist < best_dist:
                best_dist = dist
                best_surface = surface_type

        return best_surface, best_dist

    def get_floor_corners(self) -> np.ndarray:
        """Return the 4 floor corner points in image coordinates."""
        return self.image_points[:4].copy()

    def get_net_line_image(self) -> tuple[np.ndarray, np.ndarray]:
        """Estimate net position in image coordinates.

        The net is at y = COURT_LENGTH/2 on the floor plane.
        We interpolate between the floor corners.
        """
        fl, fr, nr, nl = self.image_points[:4]
        net_left = (fl + nl) / 2
        net_right = (fr + nr) / 2
        return net_left, net_right

    def is_ambiguous_zone(self, court_point: np.ndarray) -> bool:
        """Check if a court point lies in the boundary margin where
        floor/wall classification is unreliable due to monocular depth ambiguity.

        The ambiguous zone is the strip within ``floor_margin`` meters of
        any court edge — where a ball on the floor and a ball on the wall
        at low height project to nearly the same image location.
        """
        if not self._is_built:
            raise RuntimeError("Court geometry not built. Call build() first.")

        cx, cy = float(court_point[0]), float(court_point[1])
        margin = 0.5  # same as floor_margin in classify_by_court_coords

        near_x_edge = cx < margin or cx > COURT_WIDTH - margin
        near_y_edge = cy < margin or cy > COURT_LENGTH - margin

        well_inside = (
            margin <= cx <= COURT_WIDTH - margin
            and margin <= cy <= COURT_LENGTH - margin
        )

        return (near_x_edge or near_y_edge) and not well_inside

    def is_wall_or_glass(self, surface: Surface) -> bool:
        return surface in (
            Surface.WALL_FAR, Surface.WALL_NEAR,
            Surface.WALL_LEFT, Surface.WALL_RIGHT,
        )

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        poly = polygon.astype(np.float32).reshape(-1, 1, 2)
        result = cv2.pointPolygonTest(poly, tuple(point.astype(float)), False)
        return result >= 0
