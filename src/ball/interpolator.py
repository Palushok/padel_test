from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


class BallInterpolator:
    """Interpolates missing ball positions using observed detections."""

    def __init__(self, cfg: dict):
        self.max_gap = cfg["max_gap"]
        self.method = cfg["method"]

    def interpolate(
        self,
        detections: list,
        total_frames: int,
    ) -> list[np.ndarray | None]:
        """Fill in missing ball positions via interpolation.

        Args:
            detections: list of BallDetection (or None), indexed by frame
            total_frames: total frame count

        Returns:
            list of (x, y) positions or None per frame
        """
        observed_frames = []
        observed_x = []
        observed_y = []

        for d in detections:
            if d is not None:
                observed_frames.append(d.frame_idx)
                observed_x.append(d.position[0])
                observed_y.append(d.position[1])

        if len(observed_frames) < 2:
            result = [None] * total_frames
            for d in detections:
                if d is not None:
                    result[d.frame_idx] = d.position.copy()
            return result

        observed_frames = np.array(observed_frames)
        observed_x = np.array(observed_x)
        observed_y = np.array(observed_y)

        kind = self.method if len(observed_frames) >= 4 else "linear"
        interp_x = interp1d(
            observed_frames, observed_x, kind=kind,
            fill_value="extrapolate", bounds_error=False,
        )
        interp_y = interp1d(
            observed_frames, observed_y, kind=kind,
            fill_value="extrapolate", bounds_error=False,
        )

        result: list[np.ndarray | None] = [None] * total_frames

        for d in detections:
            if d is not None:
                result[d.frame_idx] = d.position.copy()

        for i in range(total_frames):
            if result[i] is not None:
                continue

            nearest_before = observed_frames[observed_frames <= i]
            nearest_after = observed_frames[observed_frames >= i]

            if len(nearest_before) == 0 or len(nearest_after) == 0:
                continue

            gap_before = i - nearest_before[-1]
            gap_after = nearest_after[0] - i

            if gap_before <= self.max_gap and gap_after <= self.max_gap:
                result[i] = np.array([
                    float(interp_x(i)),
                    float(interp_y(i)),
                ])

        return result
