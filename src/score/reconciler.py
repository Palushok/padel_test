from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.events.high_level import HighLevelEvent
    from src.score.ocr import ScoreReading

from src.events.high_level import HighLevelEventType

logger = logging.getLogger(__name__)


@dataclass
class ScoreChange:
    """A detected score change between two OCR readings."""
    frame_start: int
    frame_end: int
    team_scored: int  # 0 or 1 (which team gained a point/game)
    score_before: list[int]  # [team_a, team_b] rightmost column
    score_after: list[int]   # [team_a, team_b] rightmost column


@dataclass
class ReconciliationResult:
    """Result of reconciling OCR scores with FSM events."""
    corrected_events: list[HighLevelEvent]
    ocr_changes: list[ScoreChange]
    fsm_points_confirmed: int = 0
    fsm_points_removed: int = 0
    ocr_points_inserted: int = 0
    corrections_log: list[str] = field(default_factory=list)


class ScoreReconciler:
    """Reconciles FSM-predicted events with OCR-observed score.

    OCR score is treated as ground truth. When FSM and OCR disagree:
    - FSM says point scored, OCR says no change → remove FSM point
    - OCR says score changed, FSM has no point → insert point event
    - FSM point matches OCR direction → confirmed

    The reconciler preserves all non-scoring events (serves, hits, bounces)
    and only adjusts POINT_WON / GAME_OVER / SET_OVER / MATCH_OVER events.
    """

    def __init__(self, cfg: dict | None = None, fps: float = 30.0):
        self.cfg = cfg or {}
        self.fps = fps
        # How many seconds after a FSM event we wait for the scoreboard to update
        self.confirmation_delay_sec = self.cfg.get("confirmation_delay_sec", 3.0)
        self.confirmation_delay_frames = int(self.confirmation_delay_sec * self.fps)

    def reconcile(
        self,
        high_level_events: list[HighLevelEvent],
        ocr_readings: list[ScoreReading],
    ) -> ReconciliationResult:
        """Reconcile FSM events with OCR score readings.

        Args:
            high_level_events: events from GameStateMachine
            ocr_readings: valid OCR readings sorted by frame_idx

        Returns:
            ReconciliationResult with corrected events
        """
        if not ocr_readings or len(ocr_readings) < 2:
            logger.info("  Reconciler: not enough OCR readings, keeping FSM events as-is")
            return ReconciliationResult(
                corrected_events=list(high_level_events),
                ocr_changes=[],
            )

        # 1. Detect score changes from OCR timeline
        ocr_changes = self._detect_ocr_changes(ocr_readings)
        logger.info(f"  Reconciler: {len(ocr_changes)} score changes detected by OCR")

        # 2. Extract FSM point events
        fsm_points = [
            e for e in high_level_events
            if e.event_type == HighLevelEventType.POINT_WON
        ]
        logger.info(f"  Reconciler: {len(fsm_points)} points predicted by FSM")

        # 3. Match FSM points with OCR changes
        matched_fsm, matched_ocr, unmatched_fsm, unmatched_ocr = self._match(
            fsm_points, ocr_changes
        )

        # 4. Build corrected event list
        result = ReconciliationResult(
            corrected_events=[],
            ocr_changes=ocr_changes,
        )

        # Keep all non-point-cascade events
        point_cascade_types = {
            HighLevelEventType.POINT_WON,
            HighLevelEventType.GAME_OVER,
            HighLevelEventType.SET_OVER,
            HighLevelEventType.MATCH_OVER,
            HighLevelEventType.SIDE_SWITCH,
            HighLevelEventType.DOUBLE_BOUNCE,
            HighLevelEventType.NET_FAULT,
        }

        events_to_remove = set()
        for fsm_evt in unmatched_fsm:
            events_to_remove.add(id(fsm_evt))
            # Also remove the cascade events (game_over, etc.) that follow this point
            idx = high_level_events.index(fsm_evt)
            for following in high_level_events[idx + 1:]:
                if following.event_type in point_cascade_types:
                    events_to_remove.add(id(following))
                else:
                    break

        for evt in high_level_events:
            if id(evt) in events_to_remove:
                result.corrections_log.append(
                    f"REMOVED frame {evt.frame_idx}: {evt.event_type.value} "
                    f"(no OCR confirmation)"
                )
                continue
            result.corrected_events.append(evt)

        result.fsm_points_removed = len(unmatched_fsm)

        # 5. Insert OCR-detected points that FSM missed
        for ocr_change in unmatched_ocr:
            from src.events.high_level import HighLevelEvent as HLE
            insert_frame = (ocr_change.frame_start + ocr_change.frame_end) // 2

            new_event = HLE(
                event_type=HighLevelEventType.POINT_WON,
                frame_idx=insert_frame,
                details={
                    "winning_team": ocr_change.team_scored,
                    "reason": "ocr_detected",
                    "score_before": f"{ocr_change.score_before}",
                    "score_after": f"{ocr_change.score_after}",
                    "source": "ocr",
                },
            )
            result.corrected_events.append(new_event)
            result.corrections_log.append(
                f"INSERTED frame {insert_frame}: point_won team={ocr_change.team_scored} "
                f"(OCR: {ocr_change.score_before} → {ocr_change.score_after})"
            )

        result.corrected_events.sort(key=lambda e: e.frame_idx)

        result.fsm_points_confirmed = len(matched_fsm)
        result.ocr_points_inserted = len(unmatched_ocr)

        for entry in result.corrections_log:
            logger.info(f"    {entry}")

        logger.info(
            f"  Reconciler: {result.fsm_points_confirmed} confirmed, "
            f"{result.fsm_points_removed} removed, "
            f"{result.ocr_points_inserted} inserted"
        )

        return result

    def _detect_ocr_changes(
        self, readings: list[ScoreReading]
    ) -> list[ScoreChange]:
        """Detect score changes between consecutive OCR readings."""
        changes: list[ScoreChange] = []

        for i in range(1, len(readings)):
            prev = readings[i - 1]
            curr = readings[i]

            if not prev.team_a_scores or not curr.team_a_scores:
                continue
            if not prev.team_b_scores or not curr.team_b_scores:
                continue

            # Compare rightmost score column (usually current games or points)
            prev_a = prev.team_a_scores[-1]
            prev_b = prev.team_b_scores[-1]
            curr_a = curr.team_a_scores[-1]
            curr_b = curr.team_b_scores[-1]

            da = curr_a - prev_a
            db = curr_b - prev_b

            if da > 0 and db == 0:
                changes.append(ScoreChange(
                    frame_start=prev.frame_idx,
                    frame_end=curr.frame_idx,
                    team_scored=0,
                    score_before=[prev_a, prev_b],
                    score_after=[curr_a, curr_b],
                ))
            elif db > 0 and da == 0:
                changes.append(ScoreChange(
                    frame_start=prev.frame_idx,
                    frame_end=curr.frame_idx,
                    team_scored=1,
                    score_before=[prev_a, prev_b],
                    score_after=[curr_a, curr_b],
                ))
            elif da > 0 and db > 0:
                # Both changed — unusual, might be a set boundary reset
                # Log but don't create change events
                logger.debug(
                    f"  OCR: both scores changed between frames "
                    f"{prev.frame_idx}-{curr.frame_idx}: "
                    f"[{prev_a},{prev_b}] → [{curr_a},{curr_b}]"
                )
            elif da < 0 or db < 0:
                # Score went down — likely a new set/game reset
                # The display format changed (e.g., new set column appeared)
                logger.debug(
                    f"  OCR: score decreased between frames "
                    f"{prev.frame_idx}-{curr.frame_idx}: "
                    f"[{prev_a},{prev_b}] → [{curr_a},{curr_b}] (set change?)"
                )

        return changes

    def _match(
        self,
        fsm_points: list[HighLevelEvent],
        ocr_changes: list[ScoreChange],
    ) -> tuple[
        list[HighLevelEvent],   # matched FSM
        list[ScoreChange],       # matched OCR
        list[HighLevelEvent],   # unmatched FSM (false positives)
        list[ScoreChange],       # unmatched OCR (missed by FSM)
    ]:
        """Match FSM point events with OCR score changes.

        The scoreboard updates *after* the real event, so we use an
        asymmetric window: the FSM event should precede the OCR change
        by at most ``confirmation_delay_frames``.

        A match requires:
        1. The FSM point happened no later than ``confirmation_delay_frames``
           after the OCR change appeared (``frame_end``), and no earlier than
           ``confirmation_delay_frames`` before the OCR change window start.
        2. The winning team matches.
        """
        used_fsm: set[int] = set()
        used_ocr: set[int] = set()

        matched_fsm: list[HighLevelEvent] = []
        matched_ocr: list[ScoreChange] = []

        delay = self.confirmation_delay_frames

        for oi, ocr_change in enumerate(ocr_changes):
            best_fsm_idx = None
            best_distance = float("inf")

            for fi, fsm_evt in enumerate(fsm_points):
                if fi in used_fsm:
                    continue

                winning_team = fsm_evt.details.get("winning_team")
                if winning_team != ocr_change.team_scored:
                    continue

                # The FSM event typically happens *before* the scoreboard
                # updates, so the main tolerance is backward from OCR.
                earliest = ocr_change.frame_start - delay
                latest = ocr_change.frame_end + delay

                if earliest <= fsm_evt.frame_idx <= latest:
                    dist = abs(
                        fsm_evt.frame_idx
                        - (ocr_change.frame_start + ocr_change.frame_end) / 2
                    )
                    if dist < best_distance:
                        best_distance = dist
                        best_fsm_idx = fi

            if best_fsm_idx is not None:
                used_fsm.add(best_fsm_idx)
                used_ocr.add(oi)
                matched_fsm.append(fsm_points[best_fsm_idx])
                matched_ocr.append(ocr_change)

        unmatched_fsm = [
            e for i, e in enumerate(fsm_points) if i not in used_fsm
        ]
        unmatched_ocr = [
            c for i, c in enumerate(ocr_changes) if i not in used_ocr
        ]

        return matched_fsm, matched_ocr, unmatched_fsm, unmatched_ocr
