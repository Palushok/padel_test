from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.court.homography import HomographyTransformer
    from src.events.low_level import LowLevelEvent
    from src.players.tracker import FrameTracking


TEAM_COLORS = {
    0: (255, 100, 50),
    1: (50, 100, 255),
    -1: (200, 200, 200),
}

BALL_COLOR = (0, 255, 255)
TRAIL_COLOR = (0, 200, 200)
COURT_COLOR = (0, 255, 0)
NET_COLOR = (255, 255, 255)
RACKET_COLOR = (0, 200, 0)

EVENT_COLORS = {
    "floor_bounce": (0, 255, 0),
    "wall_bounce": (0, 165, 255),
    "racket_hit": (255, 0, 255),
    "net_hit": (0, 0, 255),
}


class Renderer:
    def __init__(self, cfg: dict, homography: HomographyTransformer):
        self.cfg = cfg
        self.homography = homography
        self.trail_length = cfg.get("trail_length", 15)
        self.minimap_size = tuple(cfg.get("minimap_size", [300, 600]))
        self.minimap_pos = cfg.get("minimap_position", "top-right")
        self.draw_court = cfg.get("draw_court", True)
        self.draw_players = cfg.get("draw_players", True)
        self.draw_ball = cfg.get("draw_ball", True)
        self.draw_rackets = cfg.get("draw_rackets", True)
        self.draw_trail = cfg.get("draw_trail", True)
        self.draw_minimap = cfg.get("draw_minimap", True)

    def render_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        court_corners: np.ndarray | None,
        tracking: FrameTracking | None,
        ball_position: np.ndarray | None,
        ball_trail: list[np.ndarray | None],
        events: list[LowLevelEvent],
        team_assignments: dict[int, int],
        score_text: str = "",
    ) -> np.ndarray:
        out = frame.copy()

        if self.draw_court and court_corners is not None:
            self._draw_court(out, court_corners)

        if self.draw_trail and ball_trail:
            self._draw_ball_trail(out, ball_trail)

        if self.draw_ball and ball_position is not None:
            self._draw_ball(out, ball_position)

        if self.draw_players and tracking is not None:
            self._draw_players(out, tracking, team_assignments)

        if self.draw_rackets and tracking is not None:
            self._draw_rackets(out, tracking, team_assignments)

        self._draw_events(out, events, frame_idx)

        if score_text:
            self._draw_score(out, score_text)

        if self.draw_minimap:
            minimap = self._render_minimap(tracking, ball_position, team_assignments)
            out = self._overlay_minimap(out, minimap)

        return out

    def _draw_court(self, frame: np.ndarray, corners: np.ndarray) -> None:
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=COURT_COLOR, thickness=2)

        net_left = self.homography.court_to_image(
            np.array([0, self.homography.court_length / 2])
        )
        net_right = self.homography.court_to_image(
            np.array([self.homography.court_width, self.homography.court_length / 2])
        )
        cv2.line(
            frame,
            tuple(net_left.astype(int)),
            tuple(net_right.astype(int)),
            NET_COLOR, 2,
        )

    def _draw_players(
        self, frame: np.ndarray, tracking: FrameTracking,
        team_assignments: dict[int, int],
    ) -> None:
        for player in tracking.players:
            team_id = team_assignments.get(player.track_id, -1)
            color = TEAM_COLORS.get(team_id, TEAM_COLORS[-1])
            x1, y1, x2, y2 = player.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"P{player.track_id} T{team_id}"
            cv2.putText(
                frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

    def _draw_rackets(
        self, frame: np.ndarray, tracking: FrameTracking,
        team_assignments: dict[int, int],
    ) -> None:
        for racket in tracking.rackets:
            x1, y1, x2, y2 = racket.bbox.astype(int)
            owner_team = team_assignments.get(racket.owner_track_id, -1) if racket.owner_track_id else -1
            color = TEAM_COLORS.get(owner_team, RACKET_COLOR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            label = f"R(P{racket.owner_track_id})" if racket.owner_track_id else "R"
            cv2.putText(
                frame, label, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

    def _draw_ball(self, frame: np.ndarray, position: np.ndarray) -> None:
        pos = tuple(position.astype(int))
        cv2.circle(frame, pos, 8, BALL_COLOR, -1)
        cv2.circle(frame, pos, 10, (0, 0, 0), 2)

    def _draw_ball_trail(
        self, frame: np.ndarray, trail: list[np.ndarray | None]
    ) -> None:
        recent = trail[-self.trail_length:]
        for i, pos in enumerate(recent):
            if pos is None:
                continue
            alpha = (i + 1) / len(recent)
            radius = max(2, int(6 * alpha))
            color = tuple(int(c * alpha) for c in TRAIL_COLOR)
            cv2.circle(frame, tuple(pos.astype(int)), radius, color, -1)

    def _draw_events(
        self, frame: np.ndarray, events: list[LowLevelEvent], frame_idx: int
    ) -> None:
        for event in events:
            if abs(event.frame_idx - frame_idx) > 15:
                continue
            color = EVENT_COLORS.get(event.event_type.value, (255, 255, 255))
            pos = tuple(event.ball_position_image.astype(int))
            cv2.circle(frame, pos, 15, color, 3)

            label = event.event_type.value.replace("_", " ").upper()
            if event.action_type:
                label += f" ({event.action_type})"
            elif event.surface:
                label += f" ({event.surface})"
            cv2.putText(
                frame, label, (pos[0] + 18, pos[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2,
            )

    def _draw_score(self, frame: np.ndarray, text: str) -> None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, h - 50), (500, h - 10), (0, 0, 0), -1)
        cv2.putText(
            frame, text, (15, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

    def _render_minimap(
        self, tracking: FrameTracking | None,
        ball_position: np.ndarray | None,
        team_assignments: dict[int, int],
    ) -> np.ndarray:
        mw, mh = self.minimap_size
        minimap = np.zeros((mh, mw, 3), dtype=np.uint8)

        cv2.rectangle(minimap, (5, 5), (mw - 5, mh - 5), (40, 80, 40), -1)
        cv2.rectangle(minimap, (5, 5), (mw - 5, mh - 5), (255, 255, 255), 2)

        net_y = mh // 2
        cv2.line(minimap, (5, net_y), (mw - 5, net_y), NET_COLOR, 2)

        service_y_top = int(mh * 3 / 20)
        service_y_bot = int(mh * 17 / 20)
        cv2.line(minimap, (5, service_y_top), (mw - 5, service_y_top), (200, 200, 200), 1)
        cv2.line(minimap, (5, service_y_bot), (mw - 5, service_y_bot), (200, 200, 200), 1)
        cv2.line(minimap, (mw // 2, service_y_top), (mw // 2, service_y_bot), (200, 200, 200), 1)

        def court_to_minimap(court_pt: np.ndarray) -> tuple[int, int]:
            x_ratio = court_pt[0] / self.homography.court_width
            y_ratio = court_pt[1] / self.homography.court_length
            return (int(5 + x_ratio * (mw - 10)), int(5 + y_ratio * (mh - 10)))

        if tracking is not None:
            for player in tracking.players:
                court_pos = self.homography.image_to_court(player.foot_position)
                mpos = court_to_minimap(court_pos)
                team_id = team_assignments.get(player.track_id, -1)
                color = TEAM_COLORS.get(team_id, TEAM_COLORS[-1])
                cv2.circle(minimap, mpos, 10, color, -1)
                cv2.putText(
                    minimap, str(player.track_id), (mpos[0] - 5, mpos[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                )

        if ball_position is not None:
            court_ball = self.homography.image_to_court(ball_position)
            bpos = court_to_minimap(court_ball)
            cv2.circle(minimap, bpos, 6, BALL_COLOR, -1)

        return minimap

    def _overlay_minimap(
        self, frame: np.ndarray, minimap: np.ndarray
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        mh, mw = minimap.shape[:2]
        margin = 10

        if self.minimap_pos == "top-right":
            y1, x1 = margin, w - mw - margin
        elif self.minimap_pos == "top-left":
            y1, x1 = margin, margin
        elif self.minimap_pos == "bottom-right":
            y1, x1 = h - mh - margin, w - mw - margin
        else:
            y1, x1 = h - mh - margin, margin

        frame[y1:y1 + mh, x1:x1 + mw] = minimap
        return frame
