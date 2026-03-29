from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.ball.kalman_filter import FilteredBallState
    from src.court.geometry import CourtGeometry, Surface
    from src.court.homography import HomographyTransformer
    from src.events.low_level import LowLevelEvent, LowLevelEventType


@dataclass
class BounceAnalysis:
    """Result of trajectory-based surface disambiguation."""
    surface: str           # Surface enum value
    p_floor: float         # probability that this is a floor bounce
    confidence: float      # abs(p_floor - 0.5) * 2, in [0, 1]
    signals: dict          # individual signal values for debugging


class TrajectoryAnalyzer:
    """Analyzes ball trajectory segments to disambiguate floor vs. wall bounces.

    Uses four independent signals:
    A) Padel rule-based prior (event sequence)
    B) Post-bounce court-Y displacement
    C) Image-space trajectory curvature
    D) Rally state continuity
    """

    def __init__(self, cfg: dict, homography: HomographyTransformer):
        self.homography = homography
        self.post_window = cfg.get("post_bounce_window", 12)
        self.pre_window = cfg.get("pre_bounce_window", 10)
        self.court_y_threshold = cfg.get("court_y_displacement_threshold", 1.5)
        self.curvature_threshold = cfg.get("curvature_threshold", 0.01)

        weights = cfg.get("weights", {})
        self.w_rules = weights.get("rules", 0.35)
        self.w_court_y = weights.get("court_y", 0.25)
        self.w_curvature = weights.get("curvature", 0.20)
        self.w_rally = weights.get("rally", 0.20)

    def analyze_bounce(
        self,
        frame_idx: int,
        ball_states: list[FilteredBallState],
        prior_events: list[LowLevelEvent],
        rally_active: bool | None,
    ) -> BounceAnalysis:
        """Determine whether an ambiguous velocity change is floor or wall bounce.

        Args:
            frame_idx: frame where the velocity change was detected
            ball_states: full Kalman-filtered ball trajectory
            prior_events: low-level events detected so far (before this frame)
            rally_active: rally state at this frame (None if unavailable)

        Returns:
            BounceAnalysis with surface classification and confidence.
        """
        from src.court.geometry import Surface

        p_rules = self._signal_rules(frame_idx, prior_events)
        p_court_y = self._signal_court_y(frame_idx, ball_states)
        p_curvature = self._signal_curvature(frame_idx, ball_states)
        p_rally = self._signal_rally(frame_idx, prior_events, rally_active)

        p_floor = (
            self.w_rules * p_rules
            + self.w_court_y * p_court_y
            + self.w_curvature * p_curvature
            + self.w_rally * p_rally
        )

        surface = Surface.FLOOR if p_floor > 0.5 else Surface.WALL_NEAR
        confidence = abs(p_floor - 0.5) * 2.0

        if p_floor <= 0.5:
            surface = self._infer_wall_direction(frame_idx, ball_states)

        return BounceAnalysis(
            surface=surface.value,
            p_floor=p_floor,
            confidence=confidence,
            signals={
                "p_rules": p_rules,
                "p_court_y": p_court_y,
                "p_curvature": p_curvature,
                "p_rally": p_rally,
            },
        )

    def _signal_rules(
        self, frame_idx: int, prior_events: list[LowLevelEvent]
    ) -> float:
        """Signal A: padel rule-based prior from event sequence.

        After RACKET_HIT the ball must bounce on the floor first (legal play).
        First contact → high P(floor). Second contact after floor → high P(wall).
        """
        from src.events.low_level import LowLevelEventType

        last_hit_idx = None
        contacts_since_hit: list[LowLevelEvent] = []

        for evt in reversed(prior_events):
            if evt.frame_idx >= frame_idx:
                continue
            if evt.event_type == LowLevelEventType.RACKET_HIT:
                last_hit_idx = evt.frame_idx
                break
            if evt.event_type in (
                LowLevelEventType.FLOOR_BOUNCE,
                LowLevelEventType.WALL_BOUNCE,
            ):
                contacts_since_hit.append(evt)

        if last_hit_idx is None:
            return 0.5

        contacts_since_hit.reverse()
        n_contacts = len(contacts_since_hit)

        if n_contacts == 0:
            return 0.85
        if n_contacts >= 1:
            first_was_floor = (
                contacts_since_hit[0].event_type == LowLevelEventType.FLOOR_BOUNCE
            )
            if first_was_floor:
                return 0.2
        return 0.5

    def _signal_court_y(
        self, frame_idx: int, ball_states: list[FilteredBallState]
    ) -> float:
        """Signal B: post-bounce court-Y displacement.

        Wall bounce → ball reverses toward court center (large court-Y change).
        Floor bounce → ball goes up, smaller court-Y change initially.
        """
        total = len(ball_states)
        end = min(frame_idx + self.post_window, total - 1)

        if end <= frame_idx:
            return 0.5

        pos_at_bounce = ball_states[frame_idx].position
        pos_after = ball_states[end].position

        court_bounce = self.homography.image_to_court(pos_at_bounce)
        court_after = self.homography.image_to_court(pos_after)

        delta_y = abs(float(court_after[1]) - float(court_bounce[1]))

        if delta_y > self.court_y_threshold:
            return 0.2
        return 0.7

    def _signal_curvature(
        self, frame_idx: int, ball_states: list[FilteredBallState]
    ) -> float:
        """Signal C: image-space trajectory curvature after the bounce.

        Floor bounce → pronounced parabolic arc (high curvature).
        Wall bounce → flatter image trajectory (low curvature).
        """
        total = len(ball_states)
        start = frame_idx
        end = min(frame_idx + self.post_window, total)

        positions = []
        for i in range(start, end):
            if ball_states[i].is_observed:
                positions.append(ball_states[i].position.copy())

        if len(positions) < 5:
            return 0.5

        xs = np.array([p[0] for p in positions])
        ys = np.array([p[1] for p in positions])
        ts = np.arange(len(positions), dtype=np.float64)

        try:
            coeffs_y = np.polyfit(ts, ys, 2)
            curvature = abs(coeffs_y[0])
        except (np.linalg.LinAlgError, ValueError):
            return 0.5

        if curvature > self.curvature_threshold:
            return 0.75
        return 0.3

    def _signal_rally(
        self,
        frame_idx: int,
        prior_events: list[LowLevelEvent],
        rally_active: bool | None,
    ) -> float:
        """Signal D: rally state continuity.

        If rally is active, the contact sequence is legal → first contact
        after a hit should be floor bounce.
        """
        from src.events.low_level import LowLevelEventType

        if rally_active is None:
            return 0.5

        last_hit = None
        floor_since_hit = False
        for evt in reversed(prior_events):
            if evt.frame_idx >= frame_idx:
                continue
            if evt.event_type == LowLevelEventType.RACKET_HIT:
                last_hit = evt
                break
            if evt.event_type == LowLevelEventType.FLOOR_BOUNCE:
                floor_since_hit = True

        if rally_active:
            if last_hit is not None and not floor_since_hit:
                return 0.9
            if floor_since_hit:
                return 0.3
            return 0.6
        return 0.5

    def _infer_wall_direction(
        self, frame_idx: int, ball_states: list[FilteredBallState]
    ) -> Surface:
        """When the bounce is classified as wall, determine which wall.

        Uses the ball's court coordinates: the wall closest to the ball
        in the direction of travel.
        """
        from src.court.geometry import COURT_LENGTH, COURT_WIDTH, Surface

        pos = ball_states[frame_idx].position
        court = self.homography.image_to_court(pos)
        cx, cy = float(court[0]), float(court[1])

        distances = {
            Surface.WALL_NEAR: abs(cy - COURT_LENGTH),
            Surface.WALL_FAR: abs(cy),
            Surface.WALL_LEFT: abs(cx),
            Surface.WALL_RIGHT: abs(cx - COURT_WIDTH),
        }

        return min(distances, key=distances.get)
