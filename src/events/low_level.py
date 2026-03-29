from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.ball.kalman_filter import FilteredBallState
    from src.ball.trajectory_analyzer import TrajectoryAnalyzer
    from src.court.geometry import CourtGeometry
    from src.court.homography import HomographyTransformer
    from src.players.action_recognition import ActionPrediction
    from src.players.rally_classifier import RallyState
    from src.players.tracker import FrameTracking


class LowLevelEventType(Enum):
    FLOOR_BOUNCE = "floor_bounce"
    WALL_BOUNCE = "wall_bounce"
    GLASS_BOUNCE = "glass_bounce"
    RACKET_HIT = "racket_hit"
    NET_HIT = "net_hit"


@dataclass
class LowLevelEvent:
    event_type: LowLevelEventType
    frame_idx: int
    ball_position_image: np.ndarray
    ball_position_court: np.ndarray | None
    player_track_id: int | None = None
    court_side: str | None = None       # 'near' or 'far'
    surface: str | None = None          # from CourtGeometry.Surface
    action_type: str | None = None      # from ActionRecognizer if available
    confidence: float = 1.0
    details: dict | None = None


class LowLevelEventDetector:
    """Detects low-level ball events using 3D court geometry.

    Uses CourtGeometry to classify which surface the ball is on,
    and optionally uses action recognition predictions to confirm racket hits.

    Detection strategy:
    1. Find frames where ball velocity changes sharply.
    2. Classify using court geometry:
       - Ball near a detected racket → RACKET_HIT (preferred)
       - Ball near a player (fallback when no racket detected) → RACKET_HIT
       - Ball on Floor surface → FLOOR_BOUNCE
       - Ball on Wall surface (far/near/left/right) → WALL_BOUNCE
       - Near net and speed drops → NET_HIT
    """

    def __init__(
        self,
        cfg: dict,
        homography: HomographyTransformer,
        court_geometry: CourtGeometry,
        trajectory_analyzer: TrajectoryAnalyzer | None = None,
        rally_states: list[RallyState] | None = None,
    ):
        self.cfg = cfg
        self.homography = homography
        self.geometry = court_geometry
        self.trajectory_analyzer = trajectory_analyzer
        self._rally_map: dict[int, bool] = {}
        if rally_states:
            self._rally_map = {rs.frame_idx: rs.is_rally_active for rs in rally_states}
        self.vel_threshold = cfg["velocity_change_threshold"]
        self.racket_hit_distance = cfg["racket_hit_distance"]
        self.player_hit_distance = cfg["player_hit_distance"]
        self.net_zone_w = cfg["net_zone_width"]
        self.net_speed_drop = cfg["net_speed_drop_ratio"]

    def detect(
        self,
        ball_states: list[FilteredBallState],
        frame_trackings: list[FrameTracking],
        action_predictions: list[ActionPrediction] | None = None,
    ) -> list[LowLevelEvent]:
        change_frames = self._find_velocity_change_points(ball_states)
        events: list[LowLevelEvent] = []

        tracking_map = {ft.frame_idx: ft for ft in frame_trackings}

        action_map: dict[tuple[int, int], str] = {}
        if action_predictions:
            for ap in action_predictions:
                for f in range(ap.frame_start, ap.frame_end + 1):
                    action_map[(f, ap.track_id)] = ap.action.value

        for fidx in change_frames:
            state = ball_states[fidx]
            if not state.is_observed and fidx > 0 and not ball_states[fidx - 1].is_observed:
                continue

            ball_img = state.position
            ball_court = self.homography.image_to_court(ball_img)
            ft = tracking_map.get(fidx)

            event = self._classify_change_point(
                fidx, ball_img, ball_court, state, ball_states, ft, action_map, events
            )
            if event is not None:
                events.append(event)

        return events

    def _find_velocity_change_points(
        self, ball_states: list[FilteredBallState]
    ) -> list[int]:
        change_frames = []

        for i in range(2, len(ball_states) - 1):
            s_prev = ball_states[i - 1]
            s_curr = ball_states[i]

            if not s_curr.is_observed and not s_prev.is_observed:
                continue

            vel_prev = s_prev.velocity
            vel_curr = s_curr.velocity
            speed_prev = np.linalg.norm(vel_prev)
            speed_curr = np.linalg.norm(vel_curr)

            if speed_prev < 1e-3 and speed_curr < 1e-3:
                continue

            if speed_prev > 1e-3 and speed_curr > 1e-3:
                cos_angle = np.dot(vel_prev, vel_curr) / (speed_prev * speed_curr)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_change = np.degrees(np.arccos(cos_angle))
            else:
                angle_change = 0

            accel_mag = np.linalg.norm(s_curr.acceleration)

            if angle_change > 30 or accel_mag > self.vel_threshold:
                change_frames.append(i)

        return self._suppress_nearby(change_frames, min_gap=5)

    @staticmethod
    def _suppress_nearby(frames: list[int], min_gap: int = 5) -> list[int]:
        if not frames:
            return []
        result = [frames[0]]
        for f in frames[1:]:
            if f - result[-1] >= min_gap:
                result.append(f)
        return result

    def _classify_change_point(
        self,
        fidx: int,
        ball_img: np.ndarray,
        ball_court: np.ndarray,
        state: FilteredBallState,
        all_states: list[FilteredBallState],
        frame_tracking: FrameTracking | None,
        action_map: dict[tuple[int, int], str],
        prior_events: list[LowLevelEvent] | None = None,
    ) -> LowLevelEvent | None:
        from src.court.geometry import Surface

        net_y = self.homography.court_length / 2

        # 1. Check racket hit: prefer ball-to-racket, fall back to ball-to-player
        if frame_tracking is not None:
            owner_id, racket_dist = self._nearest_racket(ball_img, frame_tracking)
            if owner_id is not None and racket_dist < self.racket_hit_distance:
                side = self.homography.get_court_side(ball_court)
                action = action_map.get((fidx, owner_id))
                return LowLevelEvent(
                    event_type=LowLevelEventType.RACKET_HIT,
                    frame_idx=fidx,
                    ball_position_image=ball_img,
                    ball_position_court=ball_court,
                    player_track_id=owner_id,
                    court_side=side,
                    action_type=action,
                    details={"racket_distance": float(racket_dist)},
                )

            closest_player, player_dist = self._nearest_player(ball_img, frame_tracking)
            if player_dist < self.player_hit_distance:
                side = self.homography.get_court_side(ball_court)
                action = action_map.get((fidx, closest_player))
                return LowLevelEvent(
                    event_type=LowLevelEventType.RACKET_HIT,
                    frame_idx=fidx,
                    ball_position_image=ball_img,
                    ball_position_court=ball_court,
                    player_track_id=closest_player,
                    court_side=side,
                    action_type=action,
                    details={"player_distance": float(player_dist)},
                )

        # 2. Check net hit
        if abs(ball_court[1] - net_y) < self.net_zone_w:
            speed_before = np.linalg.norm(all_states[max(0, fidx - 3)].velocity)
            speed_after = np.linalg.norm(state.velocity)
            if speed_before > 1e-3 and speed_after / speed_before < self.net_speed_drop:
                return LowLevelEvent(
                    event_type=LowLevelEventType.NET_HIT,
                    frame_idx=fidx,
                    ball_position_image=ball_img,
                    ball_position_court=ball_court,
                    details={"speed_ratio": float(speed_after / speed_before)},
                )

        # 3. Classify by court geometry (homography-based, handles polygon overlap)
        surface = self.geometry.classify_by_court_coords(ball_court, ball_img)
        side = self.homography.get_court_side(ball_court)

        # 3b. If in ambiguous zone, defer to trajectory analysis
        use_trajectory = (
            self.trajectory_analyzer is not None
            and self.geometry.is_ambiguous_zone(ball_court)
        )

        if use_trajectory:
            rally_active = self._rally_map.get(fidx)
            analysis = self.trajectory_analyzer.analyze_bounce(
                fidx, all_states, prior_events or [], rally_active,
            )
            from src.court.geometry import Surface as Sfc
            resolved = Sfc(analysis.surface)
            event_confidence = analysis.confidence

            if resolved == Sfc.FLOOR:
                return LowLevelEvent(
                    event_type=LowLevelEventType.FLOOR_BOUNCE,
                    frame_idx=fidx,
                    ball_position_image=ball_img,
                    ball_position_court=ball_court,
                    court_side=side,
                    surface=resolved.value,
                    confidence=event_confidence,
                    details={"trajectory_signals": analysis.signals},
                )
            if self.geometry.is_wall_or_glass(resolved):
                return LowLevelEvent(
                    event_type=LowLevelEventType.WALL_BOUNCE,
                    frame_idx=fidx,
                    ball_position_image=ball_img,
                    ball_position_court=ball_court,
                    court_side=side,
                    surface=resolved.value,
                    confidence=event_confidence,
                    details={"wall": resolved.value, "trajectory_signals": analysis.signals},
                )

        if surface == Surface.FLOOR:
            return LowLevelEvent(
                event_type=LowLevelEventType.FLOOR_BOUNCE,
                frame_idx=fidx,
                ball_position_image=ball_img,
                ball_position_court=ball_court,
                court_side=side,
                surface=surface.value,
            )

        if self.geometry.is_wall_or_glass(surface):
            return LowLevelEvent(
                event_type=LowLevelEventType.WALL_BOUNCE,
                frame_idx=fidx,
                ball_position_image=ball_img,
                ball_position_court=ball_court,
                court_side=side,
                surface=surface.value,
                details={"wall": surface.value},
            )

        return None

    @staticmethod
    def _nearest_racket(
        ball_img: np.ndarray, ft: FrameTracking
    ) -> tuple[int | None, float]:
        """Find the nearest detected racket and return its owner's track_id."""
        min_dist = float("inf")
        owner_id = None
        for racket in ft.rackets:
            dist = float(np.linalg.norm(ball_img - racket.center))
            if dist < min_dist:
                min_dist = dist
                owner_id = racket.owner_track_id
        return owner_id, min_dist

    @staticmethod
    def _nearest_player(
        ball_img: np.ndarray, ft: FrameTracking
    ) -> tuple[int | None, float]:
        min_dist = float("inf")
        closest_id = None
        for player in ft.players:
            dist = float(np.linalg.norm(ball_img - player.center))
            if dist < min_dist:
                min_dist = dist
                closest_id = player.track_id
        return closest_id, min_dist
