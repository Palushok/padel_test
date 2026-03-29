#!/usr/bin/env python3
"""CLI entry point for the padel analysis pipeline."""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import Pipeline


@click.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option(
    "--config", "-c",
    default="config/default.yaml",
    type=click.Path(exists=True),
    help="Path to config YAML file",
)
@click.option("--output-video", "-o", default=None, help="Output video path")
@click.option("--output-json", "-j", default=None, help="Output JSON path")
@click.option("--no-render", is_flag=True, default=False, help="Skip video rendering")
@click.option(
    "--floor-points", default=None,
    help='Floor corners as JSON: "[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]"',
)
@click.option(
    "--wall-top-points", default=None,
    help='Wall top corners as JSON: "[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]"',
)
@click.option(
    "--action-model", default=None,
    help="Path to action recognition model weights (enables action recognition)",
)
@click.option(
    "--rally-model", default=None,
    help="Path to rally classifier model weights (improves rally state detection over heuristic)",
)
@click.option(
    "--score-roi", default=None,
    help='Score overlay region as JSON: "[x1, y1, x2, y2]" in pixels (auto-detected if omitted)',
)
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    video_path: str,
    config: str,
    output_video: str | None,
    output_json: str | None,
    no_render: bool,
    floor_points: str | None,
    wall_top_points: str | None,
    action_model: str | None,
    rally_model: str | None,
    score_roi: str | None,
    verbose: bool,
) -> None:
    """Analyze a padel match video.

    VIDEO_PATH: path to the input video file.

    Court marking requires 8 points: 4 floor corners + 4 wall top corners.
    Provide via --floor-points and --wall-top-points, or set in config.

    OCR score reading is always active — the on-screen broadcast score is
    treated as ground truth and overrides event-based scoring when they conflict.
    Use --score-roi to specify the score overlay region if auto-detection fails.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = Pipeline(config, video_path)

    if floor_points:
        import json
        pipeline.cfg["court"]["mode"] = "manual"
        pipeline.cfg["court"]["manual_points"]["floor"] = json.loads(floor_points)

    if wall_top_points:
        import json
        pipeline.cfg["court"]["manual_points"]["wall_tops"] = json.loads(wall_top_points)

    if action_model:
        pipeline.cfg["players"]["action_recognition"]["enabled"] = True
        pipeline.cfg["players"]["action_recognition"]["model_path"] = action_model
        pipeline._init_components()

    if rally_model:
        pipeline.cfg.setdefault("rally_classifier", {})["model_path"] = rally_model
        pipeline._init_components()

    if score_roi:
        import json
        pipeline.cfg.setdefault("score_ocr", {})["roi"] = json.loads(score_roi)
        pipeline._init_components()

    if output_video:
        pipeline.cfg["visualization"]["output_path"] = output_video

    result = pipeline.run()

    if output_json:
        Pipeline.export_results(result, output_json)

    if not no_render:
        pipeline.render_output(result)

    # Summary
    click.echo("\n" + "=" * 60)
    click.echo("PADEL ANALYSIS SUMMARY")
    click.echo("=" * 60)
    click.echo(f"Total frames:        {result.total_frames}")
    click.echo(f"FPS:                 {result.fps:.1f}")
    click.echo(f"Unique players:      {len(result.team_assignments)}")
    click.echo(f"Low-level events:    {len(result.low_level_events)}")
    click.echo(f"High-level events:   {len(result.high_level_events)}")

    if result.action_predictions:
        click.echo(f"Action predictions:  {len(result.action_predictions)}")

    if result.rally_states:
        active = sum(1 for rs in result.rally_states if rs.is_rally_active)
        source = result.rally_states[0].source if result.rally_states else "n/a"
        click.echo(f"Rally active frames: {active}/{len(result.rally_states)} (source: {source})")

    if result.ocr_readings:
        click.echo(f"OCR readings:        {len(result.ocr_readings)}")

    if result.reconciliation:
        r = result.reconciliation
        click.echo(
            f"Score reconciliation: "
            f"{r.fsm_points_confirmed} confirmed, "
            f"{r.fsm_points_removed} removed, "
            f"{r.ocr_points_inserted} inserted"
        )

    if result.score:
        click.echo(f"Final score:         {result.score.to_string()}")

    click.echo()
    click.echo("Low-level event breakdown:")
    ll_counts = Counter(e.event_type.value for e in result.low_level_events)
    for event_type, count in ll_counts.most_common():
        click.echo(f"  {event_type:20s} {count}")

    if result.action_predictions:
        click.echo()
        click.echo("Action recognition breakdown:")
        action_counts = Counter(a.action.value for a in result.action_predictions)
        for action, count in action_counts.most_common():
            click.echo(f"  {action:20s} {count}")

    click.echo()
    click.echo("High-level event breakdown:")
    hl_counts = Counter(e.event_type.value for e in result.high_level_events)
    for event_type, count in hl_counts.most_common():
        click.echo(f"  {event_type:20s} {count}")

    if result.reconciliation and result.reconciliation.corrections_log:
        click.echo()
        click.echo("Score corrections:")
        for entry in result.reconciliation.corrections_log:
            click.echo(f"  {entry}")

    click.echo("=" * 60)


if __name__ == "__main__":
    main()
