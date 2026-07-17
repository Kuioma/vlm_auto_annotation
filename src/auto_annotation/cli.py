import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from auto_annotation.config.loader import load_run_config
from auto_annotation.pipeline.runner import run_annotation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-annotate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True, type=Path)
    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("--config", required=True, type=Path)
    batch_parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate plugins and inputs, then print the selected batch plan",
    )
    batch_parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-item batch progress on stderr",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run_annotation(load_run_config(args.config))
        return 0
    if args.command == "batch":
        from auto_annotation.batch.config import load_batch_config
        from auto_annotation.batch.runner import BatchProgress, run_batch

        def report_progress(progress: BatchProgress) -> None:
            prefix = (
                f"[{progress.position}/{progress.total}] "
                f"{progress.item_id}"
            )
            if progress.event == "attempt_started":
                detail = f"attempt {progress.attempt} started"
            elif progress.event == "attempt_failed":
                disposition = "; retrying" if progress.retrying else ""
                detail = (
                    f"attempt {progress.attempt} failed{disposition}: "
                    f"{progress.message}"
                )
            elif progress.event == "succeeded":
                detail = f"succeeded on attempt {progress.attempt}"
            elif progress.event == "skipped":
                detail = "skipped (committed result)"
            else:
                detail = f"failed: {progress.message}"
            print(f"{prefix}: {detail}", file=sys.stderr, flush=True)

        summary = run_batch(
            load_batch_config(args.config),
            plan_only=args.plan_only,
            progress=(
                None
                if args.plan_only or args.quiet
                else report_progress
            ),
        )
        print(summary.model_dump_json(indent=2))
        return 1 if summary.failed else 0
    raise ValueError(f"unsupported command: {args.command}")
