#!/usr/bin/env python3
"""CLI for evaluating pipeline predictions against ground truth annotations."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.report import evaluate_all, load_json, print_report


@click.command()
@click.argument("predictions", type=click.Path(exists=True))
@click.argument("ground_truth", type=click.Path(exists=True))
@click.option(
    "--event-tolerance", "-t",
    default=5, show_default=True,
    help="Temporal tolerance for event matching (frames)",
)
@click.option(
    "--ball-threshold", "-b",
    default=15.0, show_default=True,
    help="Max pixel distance for ball detection TP",
)
@click.option(
    "--iou-threshold",
    default=0.3, show_default=True,
    help="Min IoU for player tracking match",
)
@click.option(
    "--ocr-tolerance",
    default=30, show_default=True,
    help="Temporal tolerance for OCR reading matching (frames)",
)
@click.option(
    "--output-json", "-o",
    default=None,
    help="Save full report as JSON",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    predictions: str,
    ground_truth: str,
    event_tolerance: int,
    ball_threshold: float,
    iou_threshold: float,
    ocr_tolerance: int,
    output_json: str | None,
    verbose: bool,
) -> None:
    """Evaluate pipeline output against ground truth.

    PREDICTIONS: path to pipeline output JSON (from ``--output-json``).
    GROUND_TRUTH: path to ground truth annotations JSON.

    Only sections present in **both** files are evaluated.
    See docs/evaluation.md for ground truth annotation format.

    \b
    Examples:
        python scripts/evaluate.py results.json gt.json
        python scripts/evaluate.py results.json gt.json -t 10 -o report.json -v
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pred = load_json(predictions)
    gt = load_json(ground_truth)

    report = evaluate_all(
        pred, gt,
        event_tolerance=event_tolerance,
        ball_distance_threshold=ball_threshold,
        iou_threshold=iou_threshold,
        ocr_tolerance=ocr_tolerance,
    )

    print_report(report)

    if output_json:
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        click.echo(f"\nFull report saved to {output_json}")


if __name__ == "__main__":
    main()
