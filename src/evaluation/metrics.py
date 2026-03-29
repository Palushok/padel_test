"""Generic metric helpers used by all evaluation modules."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PRF:
    """Precision / Recall / F1 container."""
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class DistanceStats:
    """Positional error statistics."""
    errors: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.errors)

    @property
    def mean(self) -> float:
        return float(np.mean(self.errors)) if self.errors else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self.errors)) if self.errors else 0.0

    @property
    def p90(self) -> float:
        return float(np.percentile(self.errors, 90)) if self.errors else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.errors)) if self.errors else 0.0

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": round(self.mean, 2),
            "median": round(self.median, 2),
            "p90": round(self.p90, 2),
            "std": round(self.std, 2),
        }


def match_events_temporal(
    pred_frames: list[int],
    gt_frames: list[int],
    tolerance: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy temporal matching of predicted events to GT events.

    Each GT event is matched to the closest unmatched prediction within
    *tolerance* frames.  Matching is greedy by ascending distance.

    Returns:
        matched: list of (pred_frame, gt_frame) pairs
        unmatched_pred: prediction frames with no GT match (false positives)
        unmatched_gt: GT frames with no prediction match (false negatives)
    """
    if not pred_frames or not gt_frames:
        return [], list(pred_frames), list(gt_frames)

    candidates: list[tuple[int, int, int]] = []  # (|d|, pred_idx, gt_idx)
    for pi, pf in enumerate(pred_frames):
        for gi, gf in enumerate(gt_frames):
            d = abs(pf - gf)
            if d <= tolerance:
                candidates.append((d, pi, gi))

    candidates.sort()

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matched: list[tuple[int, int]] = []

    for _, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        matched.append((pred_frames[pi], gt_frames[gi]))
        used_pred.add(pi)
        used_gt.add(gi)

    unmatched_pred = [f for i, f in enumerate(pred_frames) if i not in used_pred]
    unmatched_gt = [f for i, f in enumerate(gt_frames) if i not in used_gt]
    return matched, unmatched_pred, unmatched_gt
