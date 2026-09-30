"""Command-line interface: ``hydra-bench {generate,evaluate,summarize}``."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .config import load_json


def _cmd_generate(args):
    from .generate import run_generate

    cfg = load_json(args.config)
    run_generate(cfg, output_dir=args.output_dir, limit=args.limit)


def _cmd_evaluate(args):
    from .evaluate import run_evaluate

    cfg = load_json(args.config)
    if args.situations_dir:
        cfg["situations_dir"] = args.situations_dir
    run_evaluate(cfg, output_dir=args.output_dir, skip_target=args.skip_target, skip_judge=args.skip_judge,
                 only_targets=args.targets, limit=args.limit)


def _cmd_summarize(args):
    from .summarize import run_summarize

    run_summarize(args.results_dir, contexts=args.contexts, exclude=args.exclude,
                  bootstrap=args.bootstrap, output_dir=args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hydra-bench", description="HYDRA: HYpothesis-Driven Reasoning Assessment")
    p.add_argument("--version", action="version", version=f"hydra-bench {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate situations with the configured generator models")
    g.add_argument("-c", "--config", required=True, help="JSON config file")
    g.add_argument("-o", "--output-dir", help="override output_dir in the config")
    g.add_argument("--limit", type=int, help="use only the first N context keywords (for testing)")
    g.set_defaults(func=_cmd_generate)

    e = sub.add_parser("evaluate", help="run target models on situations and judge them with the evaluator")
    e.add_argument("-c", "--config", required=True, help="JSON config file")
    e.add_argument("-s", "--situations-dir", help="override situations_dir in the config")
    e.add_argument("-o", "--output-dir", help="override output_dir in the config")
    e.add_argument("--skip-target", action="store_true", help="only judge existing answers (e.g. with another evaluator)")
    e.add_argument("--skip-judge", action="store_true", help="only generate target answers")
    e.add_argument("--targets", nargs="+", help="run only these target labels")
    e.add_argument("--limit", type=int, help="use only the first N situations (for testing)")
    e.set_defaults(func=_cmd_evaluate)

    s = sub.add_parser("summarize", help="aggregate evaluation results")
    s.add_argument("results_dir", help="directory with evaluation CSVs, e.g. outputs/results/evaluations/<evaluator>")
    s.add_argument("--contexts", default=None, help='context list for category mapping (default: use "category" column; "ddc45" for the bundled list)')
    s.add_argument("--exclude", nargs="+", help="target labels to exclude (e.g. small models)")
    s.add_argument("--bootstrap", type=int, default=0, help="number of bootstrap resamples for 95%% CIs of the final rate")
    s.add_argument("-o", "--output-dir", help="where to write summary CSVs (default: results_dir)")
    s.set_defaults(func=_cmd_summarize)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "botocore", "anthropic", "openai", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is cached; rerun the same command to resume.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
