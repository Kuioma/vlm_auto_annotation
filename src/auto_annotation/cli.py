import argparse
from collections.abc import Sequence
from pathlib import Path

from auto_annotation.config.loader import load_run_config
from auto_annotation.pipeline.runner import run_annotation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-annotate")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run_annotation(load_run_config(args.config))
        return 0
    raise ValueError(f"unsupported command: {args.command}")
