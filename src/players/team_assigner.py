from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.court.homography import HomographyTransformer
    from src.players.tracker import FrameTracking


class TeamAssigner:
    """Assigns players to teams based on their position on the court.

    In padel, each team always occupies one half of the court.
    Teams switch sides after every odd total number of games
    (after 1-0, 2-1, 3-2, etc.).
    """

    def __init__(self):
        # side_map[team_id] = 'far' or 'near'
        # Initial convention: team 0 starts on 'far' side (top of court)
        self.side_map: dict[int, str] = {0: "far", 1: "near"}
        self.total_games_played: int = 0

    def assign_teams(
        self,
        frame_trackings: list[FrameTracking],
        homography: HomographyTransformer,
    ) -> dict[int, int]:
        """Assign team IDs based on average court position.

        For each unique track_id, accumulate which side of the court
        (relative to the net) the player spends most time on.
        Map that side to a team using the current side_map.

        Returns:
            dict mapping track_id -> team_id (0 or 1)
        """
        half_y = homography.court_length / 2
        track_y_values: dict[int, list[float]] = {}

        for ft in frame_trackings:
            for player in ft.players:
                court_pos = homography.image_to_court(player.foot_position)
                track_y_values.setdefault(player.track_id, []).append(court_pos[1])

        far_side_team = self._team_on_side("far")
        near_side_team = self._team_on_side("near")

        assignments: dict[int, int] = {}
        for tid, y_vals in track_y_values.items():
            median_y = float(np.median(y_vals))
            if median_y < half_y:
                assignments[tid] = far_side_team
            else:
                assignments[tid] = near_side_team

        return assignments

    def notify_game_completed(self) -> None:
        """Call when a game ends to potentially trigger side switch.

        Padel rule: sides switch after every odd total game count.
        Game counts where switch happens: 1, 3, 5, 7, ...
        """
        self.total_games_played += 1
        if self.total_games_played % 2 == 1:
            self._switch_sides()

    def _switch_sides(self) -> None:
        self.side_map = {0: self.side_map[1], 1: self.side_map[0]}

    def _team_on_side(self, side: str) -> int:
        for team, s in self.side_map.items():
            if s == side:
                return team
        return -1

    def get_side_for_team(self, team_id: int) -> str:
        return self.side_map.get(team_id, "unknown")
