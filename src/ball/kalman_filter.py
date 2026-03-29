from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter


@dataclass
class FilteredBallState:
    position: np.ndarray     # [x, y] filtered position
    velocity: np.ndarray     # [vx, vy] estimated velocity
    acceleration: np.ndarray # [ax, ay] estimated acceleration
    frame_idx: int
    is_observed: bool        # True if measurement was fed (detected or interpolated)
    is_detected: bool = False  # True only if original YOLO detection existed


class BallKalmanFilter:
    """Kalman filter for ball tracking with constant-acceleration model.

    State vector: [x, y, vx, vy, ax, ay]
    Measurement: [x, y]

    Designed to run AFTER interpolation: receives a denser position list
    so velocity/acceleration estimates are more reliable.
    """

    def __init__(self, cfg: dict):
        self.process_noise = cfg["process_noise"]
        self.measurement_noise = cfg["measurement_noise"]
        self.interpolated_noise_scale = cfg.get("interpolated_noise_scale", 3.0)
        self.initial_covariance = cfg["initial_covariance"]
        self.kf: KalmanFilter | None = None
        self._base_R: np.ndarray | None = None

    def _init_filter(self, initial_pos: np.ndarray) -> None:
        self.kf = KalmanFilter(dim_x=6, dim_z=2)

        dt = 1.0

        self.kf.F = np.array([
            [1, 0, dt, 0,  0.5*dt**2, 0],
            [0, 1, 0,  dt, 0,         0.5*dt**2],
            [0, 0, 1,  0,  dt,        0],
            [0, 0, 0,  1,  0,         dt],
            [0, 0, 0,  0,  1,         0],
            [0, 0, 0,  0,  0,         1],
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ])

        self.kf.R *= self.measurement_noise
        self._base_R = self.kf.R.copy()
        self.kf.Q *= self.process_noise
        self.kf.P *= self.initial_covariance

        self.kf.x = np.array([
            initial_pos[0], initial_pos[1],
            0, 0, 0, 0
        ]).reshape(6, 1)

    def filter_trajectory(
        self,
        positions: list[np.ndarray | None],
        is_detected: list[bool],
    ) -> list[FilteredBallState]:
        """Apply Kalman filter to interpolated + detected positions.

        Args:
            positions: position per frame (detected, interpolated, or None)
            is_detected: True if position came from detector (not interpolation)

        Returns:
            list of FilteredBallState per frame
        """
        total_frames = len(positions)

        first_frame = None
        for i in range(total_frames):
            if positions[i] is not None:
                first_frame = i
                break

        if first_frame is None:
            return [
                FilteredBallState(
                    position=np.array([0.0, 0.0]),
                    velocity=np.array([0.0, 0.0]),
                    acceleration=np.array([0.0, 0.0]),
                    frame_idx=i,
                    is_observed=False,
                )
                for i in range(total_frames)
            ]

        self._init_filter(positions[first_frame])

        states: list[FilteredBallState] = []

        for i in range(total_frames):
            if i < first_frame:
                states.append(FilteredBallState(
                    position=positions[first_frame].copy(),
                    velocity=np.array([0.0, 0.0]),
                    acceleration=np.array([0.0, 0.0]),
                    frame_idx=i,
                    is_observed=False,
                ))
                continue

            self.kf.predict()

            has_measurement = positions[i] is not None
            if has_measurement:
                if is_detected[i]:
                    self.kf.R = self._base_R.copy()
                else:
                    self.kf.R = self._base_R * self.interpolated_noise_scale
                self.kf.update(positions[i].reshape(2, 1))

            state = self.kf.x.flatten()
            states.append(FilteredBallState(
                position=state[:2].copy(),
                velocity=state[2:4].copy(),
                acceleration=state[4:6].copy(),
                frame_idx=i,
                is_observed=has_measurement,
                is_detected=is_detected[i] if i < len(is_detected) else False,
            ))

        return states

    def reset(self) -> None:
        self.kf = None
        self._base_R = None
