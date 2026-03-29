from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.events.low_level import LowLevelEvent

from src.events.low_level import LowLevelEventType


class HighLevelEventType(Enum):
    SERVE = "serve"
    RALLY_HIT = "rally_hit"
    POINT_WON = "point_won"
    DOUBLE_BOUNCE = "double_bounce"
    NET_FAULT = "net_fault"
    OUT = "out"
    GAME_OVER = "game_over"
    SET_OVER = "set_over"
    MATCH_OVER = "match_over"
    SERVE_FAULT = "serve_fault"
    LET = "let"
    SIDE_SWITCH = "side_switch"


class PointReason(Enum):
    DOUBLE_BOUNCE = "double_bounce"
    NET_FAULT = "net_fault"
    OUT = "out"
    BALL_HITS_OWN_SIDE_WALL_BEFORE_CROSSING = "ball_hits_own_wall"


@dataclass
class HighLevelEvent:
    event_type: HighLevelEventType
    frame_idx: int
    details: dict = field(default_factory=dict)


class RallyState(Enum):
    WAITING_FOR_SERVE = "waiting_for_serve"
    SERVE_IN_PROGRESS = "serve_in_progress"
    RALLY = "rally"
    POINT_SCORED = "point_scored"
    DEAD_BALL = "dead_ball"


@dataclass
class Score:
    """Padel scoring: points (0,15,30,40), games, sets."""
    points: list[int] = field(default_factory=lambda: [0, 0])
    games: list[int] = field(default_factory=lambda: [0, 0])
    sets: list[list[int]] = field(default_factory=list)
    current_set: list[int] = field(default_factory=lambda: [0, 0])
    serving_team: int = 0

    POINT_MAP = {0: "0", 1: "15", 2: "30", 3: "40"}

    def point_won_by(self, team: int) -> str | None:
        """Award a point and return event if game/set/match ends."""
        self.points[team] += 1

        p0, p1 = self.points
        if p0 >= 3 and p1 >= 3:
            # Deuce rules
            if p0 - p1 >= 2:
                return self._game_won_by(0)
            elif p1 - p0 >= 2:
                return self._game_won_by(1)
            return None
        elif p0 >= 4:
            return self._game_won_by(0)
        elif p1 >= 4:
            return self._game_won_by(1)
        return None

    def _game_won_by(self, team: int) -> str:
        self.current_set[team] += 1
        self.points = [0, 0]
        self.serving_team = 1 - self.serving_team

        g0, g1 = self.current_set
        if (g0 >= 6 and g0 - g1 >= 2) or (g1 >= 6 and g1 - g0 >= 2):
            return self._set_won_by(team)
        if g0 == 7 or g1 == 7:  # tiebreak
            return self._set_won_by(team)
        return "game_over"

    def _set_won_by(self, team: int) -> str:
        self.sets.append(self.current_set.copy())
        self.current_set = [0, 0]

        sets_won = [0, 0]
        for s in self.sets:
            winner = 0 if s[0] > s[1] else 1
            sets_won[winner] += 1

        if sets_won[team] >= 2:  # best of 3
            return "match_over"
        return "set_over"

    def to_string(self) -> str:
        sets_str = " | ".join(f"{s[0]}-{s[1]}" for s in self.sets)
        current = f"{self.current_set[0]}-{self.current_set[1]}"
        pts = f"{self.POINT_MAP.get(self.points[0], self.points[0])}-{self.POINT_MAP.get(self.points[1], self.points[1])}"
        return f"Sets: [{sets_str}] Game: {current} Points: {pts}"


class GameStateMachine:
    """Processes low-level events and produces high-level game events.

    Core logic:
    - Tracks rally state and ball bounces per side
    - Detects double bounces, net faults, outs
    - Maintains score
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state = RallyState.WAITING_FOR_SERVE
        self.score = Score()

        self.events: list[HighLevelEvent] = []
        self.last_hit_side: str | None = None
        self.last_hit_team: int | None = None
        self.bounces_since_last_hit: list[dict] = []
        self.serve_bounced = False
        self.serve_faults = 0

        # Team 0 starts on 'far' side; updated on side switches
        self._team_side_map: dict[int, str] = {0: "far", 1: "near"}
        self._total_games: int = 0

    def process(
        self,
        low_level_events: list[LowLevelEvent],
        team_assignments: dict[int, int],
    ) -> list[HighLevelEvent]:
        self._team_assignments = team_assignments
        self.events = []

        for event in low_level_events:
            self._handle_event(event)

        return self.events

    def _get_team_for_player(self, track_id: int | None) -> int | None:
        if track_id is None:
            return None
        return self._team_assignments.get(track_id)

    def _handle_event(self, event: LowLevelEvent) -> None:
        if event.event_type == LowLevelEventType.RACKET_HIT:
            self._handle_racket_hit(event)
        elif event.event_type == LowLevelEventType.FLOOR_BOUNCE:
            self._handle_floor_bounce(event)
        elif event.event_type == LowLevelEventType.NET_HIT:
            self._handle_net_hit(event)
        elif event.event_type in (
            LowLevelEventType.WALL_BOUNCE, LowLevelEventType.GLASS_BOUNCE
        ):
            self._handle_wall_bounce(event)

    def _handle_racket_hit(self, event: LowLevelEvent) -> None:
        team = self._get_team_for_player(event.player_track_id)

        if self.state == RallyState.WAITING_FOR_SERVE:
            self.state = RallyState.SERVE_IN_PROGRESS
            self.serve_bounced = False
            self.bounces_since_last_hit = []
            self.last_hit_team = team
            self.last_hit_side = event.court_side
            self.events.append(HighLevelEvent(
                event_type=HighLevelEventType.SERVE,
                frame_idx=event.frame_idx,
                details={
                    "serving_team": team,
                    "player_id": event.player_track_id,
                },
            ))
            return

        if self.state in (RallyState.SERVE_IN_PROGRESS, RallyState.RALLY):
            self.state = RallyState.RALLY
            self.bounces_since_last_hit = []
            self.last_hit_team = team
            self.last_hit_side = event.court_side
            self.events.append(HighLevelEvent(
                event_type=HighLevelEventType.RALLY_HIT,
                frame_idx=event.frame_idx,
                details={
                    "team": team,
                    "player_id": event.player_track_id,
                },
            ))

    def _handle_floor_bounce(self, event: LowLevelEvent) -> None:
        if self.state == RallyState.WAITING_FOR_SERVE:
            return

        if self.state == RallyState.SERVE_IN_PROGRESS:
            if not self.serve_bounced:
                self.serve_bounced = True
                self.bounces_since_last_hit.append({
                    "frame": event.frame_idx,
                    "side": event.court_side,
                })

                hitting_team_side = self._team_side_map.get(self.last_hit_team, "")
                if event.court_side == hitting_team_side:
                    self._serve_fault(event, "serve_bounce_own_side")
                    return

                self.state = RallyState.RALLY
                return
            else:
                self.bounces_since_last_hit.append({
                    "frame": event.frame_idx,
                    "side": event.court_side,
                })

        elif self.state == RallyState.RALLY:
            self.bounces_since_last_hit.append({
                "frame": event.frame_idx,
                "side": event.court_side,
            })

        same_side_bounces = [
            b for b in self.bounces_since_last_hit
            if b["side"] == event.court_side
        ]

        if len(same_side_bounces) >= 2:
            bounce_side = event.court_side
            losing_team = self._side_to_team(bounce_side)
            winning_team = 1 - losing_team if losing_team is not None else None

            self.events.append(HighLevelEvent(
                event_type=HighLevelEventType.DOUBLE_BOUNCE,
                frame_idx=event.frame_idx,
                details={"side": bounce_side},
            ))

            self._award_point(event, winning_team, PointReason.DOUBLE_BOUNCE)

    def _handle_net_hit(self, event: LowLevelEvent) -> None:
        if self.state == RallyState.SERVE_IN_PROGRESS:
            # Could be a let (ball hits net but goes over)
            # For simplicity: if serve hits net, treat as fault
            self._serve_fault(event, "net_on_serve")
            return

        if self.state == RallyState.RALLY:
            winning_team = 1 - self.last_hit_team if self.last_hit_team is not None else None
            self.events.append(HighLevelEvent(
                event_type=HighLevelEventType.NET_FAULT,
                frame_idx=event.frame_idx,
                details={"hitting_team": self.last_hit_team},
            ))
            self._award_point(event, winning_team, PointReason.NET_FAULT)

    def _handle_wall_bounce(self, event: LowLevelEvent) -> None:
        # Wall bounces are legal in padel AFTER floor bounce on receiving side.
        # If ball hits wall on the hitting team's side before crossing net → fault.
        if self.state == RallyState.SERVE_IN_PROGRESS:
            # Serve cannot hit wall before bouncing on opponent's side
            self._serve_fault(event, "wall_before_bounce")
            return

        if self.state == RallyState.RALLY:
            hitting_team_side = self._team_side_map.get(self.last_hit_team, "")
            if event.court_side == hitting_team_side:
                winning_team = 1 - self.last_hit_team if self.last_hit_team is not None else None
                self._award_point(
                    event, winning_team,
                    PointReason.BALL_HITS_OWN_SIDE_WALL_BEFORE_CROSSING,
                )

    def _serve_fault(self, event: LowLevelEvent, reason: str) -> None:
        self.serve_faults += 1
        self.events.append(HighLevelEvent(
            event_type=HighLevelEventType.SERVE_FAULT,
            frame_idx=event.frame_idx,
            details={"reason": reason, "fault_number": self.serve_faults},
        ))

        if self.serve_faults >= 2:
            winning_team = 1 - self.last_hit_team if self.last_hit_team is not None else None
            self._award_point(event, winning_team, PointReason.NET_FAULT)
        else:
            self.state = RallyState.WAITING_FOR_SERVE
            self.bounces_since_last_hit = []

    def _award_point(
        self,
        event: LowLevelEvent,
        winning_team: int | None,
        reason: PointReason,
    ) -> None:
        if winning_team is None:
            winning_team = 0

        self.events.append(HighLevelEvent(
            event_type=HighLevelEventType.POINT_WON,
            frame_idx=event.frame_idx,
            details={
                "winning_team": winning_team,
                "reason": reason.value,
                "score_before": self.score.to_string(),
            },
        ))

        result = self.score.point_won_by(winning_team)

        if result in ("game_over", "set_over", "match_over"):
            event_type = {
                "game_over": HighLevelEventType.GAME_OVER,
                "set_over": HighLevelEventType.SET_OVER,
                "match_over": HighLevelEventType.MATCH_OVER,
            }[result]
            self.events.append(HighLevelEvent(
                event_type=event_type,
                frame_idx=event.frame_idx,
                details={"score": self.score.to_string()},
            ))

            self._total_games += 1
            # Side switch after every odd total game count (1, 3, 5, ...)
            if self._total_games % 2 == 1:
                self._team_side_map = {
                    0: self._team_side_map[1],
                    1: self._team_side_map[0],
                }
                self.events.append(HighLevelEvent(
                    event_type=HighLevelEventType.SIDE_SWITCH,
                    frame_idx=event.frame_idx,
                    details={
                        "new_sides": dict(self._team_side_map),
                        "total_games": self._total_games,
                    },
                ))

        self.state = RallyState.WAITING_FOR_SERVE
        self.bounces_since_last_hit = []
        self.serve_faults = 0

    def _side_to_team(self, side: str) -> int | None:
        for team, s in self._team_side_map.items():
            if s == side:
                return team
        return None
