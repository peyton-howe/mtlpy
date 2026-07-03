#!/usr/bin/env python3
"""Baseline performance for mtlpy operators.

Usage:
    python benchmarks/bench.py
    python benchmarks/bench.py --sizes 1024,1048576 --repeat 50
    python benchmarks/bench.py --baseline benchmarks/results/2026-07-01_abc1234.json

Each run measures, per (operator, buffer size):
  - first-dispatch latency: the very first call to that op in this process,
    which includes pipeline compile time unless the in-process PipelineCache
    or the on-disk binary archive (csrc/pipeline_cache.cpp) already has it
  - warm-dispatch latency: median of `--repeat` further calls, i.e. steady
    state with the pipeline fully cached
  - equivalent NumPy CPU timing, for context only (not a competition)

Results are saved as JSON (git commit + timestamp) to --out-dir so a later
run can be diffed against them with --baseline.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mtlpy import Device, operators

DEFAULT_SIZES = [1024, 16_384, 262_144, 4_194_304]
DEFAULT_OPS = ["add", "sub", "mul", "sqrt", "cos", "sin", "tan", "exp", "log"]

MTLPY_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "sqrt": lambda a, b: operators.sqrt(a),
    "cos": lambda a, b: operators.cos(a),
    "sin": lambda a, b: operators.sin(a),
    "tan": lambda a, b: operators.tan(a),
    "exp": lambda a, b: operators.exp(a),
    "log": lambda a, b: operators.log(a),
}
NUMPY_OPS = {
    "add": lambda a, b: np.add(a, b),
    "sub": lambda a, b: np.subtract(a, b),
    "mul": lambda a, b: np.multiply(a, b),
    "sqrt": lambda a, b: np.sqrt(a),
    "cos": lambda a, b: np.cos(a),
    "sin": lambda a, b: np.sin(a),
    "tan": lambda a, b: np.tan(a),
    "exp": lambda a, b: np.exp(a),
    "log": lambda a, b: np.log(a),
}
# bytes moved per element: reads + writes touched by the kernel
BYTES_PER_ELEMENT = {op: 12 for op in ["add", "sub", "mul"]}  # 2 reads + 1 write, f32
BYTES_PER_ELEMENT.update({op: 8 for op in ["sqrt", "cos", "sin", "tan", "exp", "log"]})  # 1 read + 1 write


def make_inputs(size: int, op: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    if op == "log":
        a = (rng.random(size, dtype=np.float32) * 10) + 0.1
    elif op == "sqrt":
        a = rng.random(size, dtype=np.float32) * 100
    else:
        a = rng.random(size, dtype=np.float32) * 10 - 5
    b = rng.random(size, dtype=np.float32) * 10 - 5
    return a, b


def bench_one(device: Device, op: str, size: int, repeat: int, warmup: int = 3) -> dict:
    a_np, b_np = make_inputs(size, op)
    a = device.buffer(a_np)
    b = device.buffer(b_np)
    fn = MTLPY_OPS[op]

    t0 = time.perf_counter()
    fn(a, b)
    first_dispatch_s = time.perf_counter() - t0

    for _ in range(warmup):
        fn(a, b)

    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(a, b)
        samples.append(time.perf_counter() - t0)
    warm_s = statistics.median(samples)

    numpy_samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        NUMPY_OPS[op](a_np, b_np)
        numpy_samples.append(time.perf_counter() - t0)
    numpy_s = statistics.median(numpy_samples)

    bytes_moved = size * BYTES_PER_ELEMENT[op]
    return {
        "op": op,
        "size": size,
        "first_dispatch_ms": first_dispatch_s * 1e3,
        "warm_dispatch_ms": warm_s * 1e3,
        "warm_elements_per_sec": size / warm_s,
        "warm_effective_gbps": (bytes_moved / warm_s) / 1e9,
        "numpy_ms": numpy_s * 1e3,
        "gpu_vs_numpy_speedup": numpy_s / warm_s,
    }


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def run_suite(sizes: list[int], ops: list[str], repeat: int) -> dict:
    device = Device()
    results = [bench_one(device, op, size, repeat) for op in ops for size in sizes]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "repeat": repeat,
        "results": results,
    }


def print_table(run: dict) -> None:
    header = f"{'op':<6} {'size':>10} {'first(ms)':>10} {'warm(ms)':>10} {'elem/s':>14} {'GB/s':>7} {'vs numpy':>9}"
    print(header)
    print("-" * len(header))
    for r in run["results"]:
        print(
            f"{r['op']:<6} {r['size']:>10} {r['first_dispatch_ms']:>10.3f} "
            f"{r['warm_dispatch_ms']:>10.4f} {r['warm_elements_per_sec']:>14,.0f} "
            f"{r['warm_effective_gbps']:>7.2f} {r['gpu_vs_numpy_speedup']:>8.2f}x"
        )


def print_comparison(run: dict, baseline: dict) -> None:
    baseline_by_key = {(r["op"], r["size"]): r for r in baseline["results"]}
    print(f"\nComparing against baseline from {baseline.get('timestamp', '?')} "
          f"(git {baseline.get('git_sha', '?')})")
    header = f"{'op':<6} {'size':>10} {'warm(ms)':>10} {'baseline(ms)':>13} {'change':>9}"
    print(header)
    print("-" * len(header))
    for r in run["results"]:
        key = (r["op"], r["size"])
        base = baseline_by_key.get(key)
        if base is None:
            continue
        pct = (r["warm_dispatch_ms"] - base["warm_dispatch_ms"]) / base["warm_dispatch_ms"] * 100
        flag = " <- REGRESSION" if pct > 10 else (" (faster)" if pct < -10 else "")
        print(
            f"{r['op']:<6} {r['size']:>10} {r['warm_dispatch_ms']:>10.4f} "
            f"{base['warm_dispatch_ms']:>13.4f} {pct:>+8.1f}%{flag}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=str, default=None,
                         help=f"comma-separated element counts (default: {DEFAULT_SIZES})")
    parser.add_argument("--ops", type=str, default=None,
                         help=f"comma-separated op names (default: {DEFAULT_OPS})")
    parser.add_argument("--repeat", type=int, default=30, help="samples per (op, size) for warm timing")
    parser.add_argument("--baseline", type=str, default=None, help="path to a previous results JSON to diff against")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "results"))
    parser.add_argument("--no-save", action="store_true", help="don't write a results JSON")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else DEFAULT_SIZES
    ops = [o.strip() for o in args.ops.split(",")] if args.ops else DEFAULT_OPS

    run = run_suite(sizes, ops, args.repeat)
    print_table(run)

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        print_comparison(run, baseline)

    if not args.no_save:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        out_path = out_dir / f"{stamp}_{run['git_sha']}.json"
        out_path.write_text(json.dumps(run, indent=2))
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
