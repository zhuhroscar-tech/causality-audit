"""Command-line interface for causality-audit."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import __version__
from .core import audit_prefix_invariance, epsilon_sweep
from .reference import build_reference_model
from .style import print_fields, resolve_style, section, status_headline


def _positive_int(flag_name: str):
    """argparse `type=` factory: parse a base-10 int and reject values < 1
    with a clean usage error, so --seq-len/--chunk-size 0 or negative fail
    at argument-parsing time (exit code 2, no traceback) instead of
    crashing deep inside numpy/reference.py with an IndexError or
    ValueError once the demo actually runs."""

    def _parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"--{flag_name} must be an integer, got {value!r}")
        if parsed < 1:
            raise argparse.ArgumentTypeError(f"--{flag_name} must be >= 1, got {parsed}")
        return parsed

    return _parse


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="causality-audit",
        description=(
            "Two-forward-pass prefix-invariance (causal-leakage) audit for "
            "sequence-model layers, per arXiv:2608.22876. Ships with a "
            "synthetic chunked-scan reference/oracle (--demo) that "
            "reproduces the real Zamba2/Nemotron-H bug class "
            "(huggingface/transformers#46741) for self-verification; does "
            "NOT audit real HuggingFace checkpoints (see README for that "
            "manual, non-CI workflow)."
        ),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--no-color", action="store_true", help="disable ANSI color output")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    p.add_argument(
        "--demo",
        choices=["clean", "buggy"],
        default=None,
        help="run the built-in synthetic reference model (clean or buggy chunked-scan) and audit it",
    )
    p.add_argument(
        "--seq-len",
        type=_positive_int("seq-len"),
        default=12,
        help="sequence length for --demo (default 12, must be >= 1)",
    )
    p.add_argument(
        "--chunk-size",
        type=_positive_int("chunk-size"),
        default=4,
        help="chunk size for --demo (default 4, must be >= 1)",
    )
    p.add_argument("--seed", type=int, default=0, help="random seed for --demo (default 0)")
    p.add_argument(
        "--epsilon-sweep",
        action="store_true",
        help="with --demo, also run the epsilon-sweep discriminator (genuine leak vs numerical floor)",
    )
    p.add_argument(
        "--check-clean",
        action="store_true",
        help="exit non-zero if the audit verdict is leak_detected (for CI use)",
    )
    return p


def _run_demo(args) -> tuple:
    rng = np.random.default_rng(args.seed)
    dim = 3
    x = rng.standard_normal((args.seq_len, dim))
    x_perturbed = x.copy()
    x_perturbed[-1] += 1.0

    buggy = args.demo == "buggy"
    layers = build_reference_model(chunk_size=args.chunk_size, decay=0.9, buggy=buggy, n_layers=2)
    result = audit_prefix_invariance(layers, x, x_perturbed, threshold=1e-9)

    sweep = None
    if args.epsilon_sweep:
        probe_index = result.first_leak_index if result.first_leak_index is not None else 0
        sweep = epsilon_sweep(
            build_layers=lambda: build_reference_model(
                chunk_size=args.chunk_size, decay=0.9, buggy=buggy, n_layers=2
            ),
            x_base=x,
            epsilons=[1e-3, 1e-2, 1e-1, 1.0],
            probe_layer_index=probe_index,
            threshold=1e-12,
        )
    return result, sweep


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    style = resolve_style(no_color_flag=args.no_color)

    if not args.demo:
        parser.print_help()
        return 0

    result, sweep = _run_demo(args)

    if args.json:
        payload = result.to_dict()
        if sweep is not None:
            payload["epsilon_sweep"] = sweep
        print(json.dumps(payload, indent=2))
    else:
        section(style.bold(f"Demo: {args.demo} chunked-scan reference model"))
        level = "fail" if result.verdict == "leak_detected" else "ok" if result.verdict == "clean" else "warn"
        print(status_headline(style, level, f"verdict: {result.verdict}"))
        print(f"  {result.note}")
        print()
        rows = [
            (layer.name, f"prefix_diff={layer.max_abs_diff_prefix:.3e}  leaked={layer.leaked}")
            for layer in result.layers
        ]
        print_fields(rows)
        if sweep is not None:
            print()
            section(style.bold("Epsilon-sweep discriminator"))
            print(f"  slope={sweep['slope']}  verdict={sweep['verdict']}")
            for eps, diff in sweep["points"]:
                print(f"    eps={eps:<8} max_abs_diff_prefix={diff:.3e}")

    if args.check_clean:
        return 1 if result.verdict == "leak_detected" else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
