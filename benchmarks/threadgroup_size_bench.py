#!/usr/bin/env python3
"""Benchmark: threadgroup shape vs GPU time for a threadgroup-memory-tiled
kernel, using Pipeline.run's threadgroup= override (see pipeline.py) to force
each candidate shape instead of relying on the auto-computed default.

gaussian_blur_tiled_variable_kernel.txt is a parametric version of
gaussian_blur_tiled_kernel.txt: TILE_X/TILE_Y are #defines (this script
prepends them) instead of a hardcoded 32, so one source is recompiled per
candidate shape and dispatched with threadgroup=(TILE_X, TILE_Y). This is
the kernel family where threadgroup size isn't just a performance knob:

  - Correctness depends on it. The shared-memory tile array is sized from
    TILE_X/TILE_Y at compile time, and the cooperative load loop assumes
    the dispatched threads-per-threadgroup matches those constants exactly.
    A mismatch (e.g. relying on the auto-computed default instead of
    threadgroup=) silently produces wrong pixels -- see
    demonstrate_mismatch_is_wrong() below, which dispatches a TILE_X=TILE_Y=32
    variant with a mismatched threadgroup=(16, 16) on purpose.
  - Performance depends on it too, independent of correctness. Tile shape
    controls how much per-thread halo-load overhead is paid to prime the
    shared-memory neighborhood (worse for skinny tiles, which load a large
    perimeter relative to their area) vs. how many threadgroups are in
    flight at once (worse for very large tiles).

Every shape's output is also checked against gaussian_blur_kernel.txt (the
untiled reference kernel) so the perf numbers can't hide a silent
correctness break the way demonstrate_mismatch_is_wrong() deliberately
triggers.

Usage:
    python benchmarks/threadgroup_size_bench.py
    python benchmarks/threadgroup_size_bench.py --sizes 1921x1081,3841x2161
    python benchmarks/threadgroup_size_bench.py --repeat 40
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

from mtlpy import Device

from demosaic_bench import warmup_until_stable

TILED_TEMPLATE_PATH = Path(__file__).parent / "gaussian_blur_tiled_variable_kernel.txt"
NAIVE_KERNEL_PATH = Path(__file__).parent / "gaussian_blur_kernel.txt"

# (width, height) per threadgroup. Deliberately spans square and skewed
# (wide/tall) shapes at several total sizes, all multiples of 32 (the
# thread_execution_width on Apple GPUs) so none are rejected outright --
# whether they're a *good* choice is exactly what this benchmark measures.
DEFAULT_SHAPES = [
    (8, 8), (16, 8), (8, 16), (16, 16),
    (32, 8), (8, 32), (32, 16), (16, 32),
    (32, 32), (64, 16), (16, 64),
    (256, 1), (1, 256),
]

# Deliberately not multiples of any candidate tile size, per this project's
# established methodology (see benchmarks/README.md's "Boundary-size
# correctness testing"): dispatchThreads truncates the last threadgroup per
# axis, which is exactly where a cooperative shared-memory load loop that
# assumes a fixed thread count would go wrong first.
DEFAULT_SIZES = [(1081, 1921), (2161, 3841)]  # (height, width)
DEFAULT_REPEAT = 20


def make_tiled_source(tile_x: int, tile_y: int) -> str:
    header = f"#define TILE_X {tile_x}\n#define TILE_Y {tile_y}\n"
    return header + TILED_TEMPLATE_PATH.read_text()


def gpu_median_ms(pipeline, buffers, grid, threadgroup, repeat: int) -> float:
    def dispatch():
        pipeline.run(buffers, grid, wait=True, threadgroup=threadgroup)

    warmup_until_stable(dispatch)

    samples = []
    for _ in range(repeat):
        gpu_start, gpu_end = pipeline.run(buffers, grid, wait=True, threadgroup=threadgroup)
        samples.append((gpu_end - gpu_start) * 1e3)
    return statistics.median(samples)


def make_source_and_reference(device: Device, naive_pipeline, height: int, width: int, seed: int):
    """Random source image plus its untiled-reference blur output -- shared
    setup for bench_size() and demonstrate_mismatch_is_wrong(), which both
    need a source buffer and a known-correct reference to diff against."""
    rng = np.random.default_rng(seed)
    src_np = rng.random((height, width)).astype(np.float32)

    buf_in = device.buffer(src_np.reshape(-1))
    buf_out_naive = device.empty(height * width, np.float32)
    width_buf = device.buffer(np.array([width], dtype=np.uint32))
    height_buf = device.buffer(np.array([height], dtype=np.uint32))
    grid = [width, height, 1]

    naive_pipeline.run([buf_in, buf_out_naive, width_buf, height_buf], grid, wait=True)
    reference = buf_out_naive.contents.reshape(height, width).copy()

    return buf_in, width_buf, height_buf, grid, reference


def bench_size(device: Device, naive_pipeline, height: int, width: int,
                shapes: list[tuple[int, int]], repeat: int) -> list[dict]:
    buf_in, width_buf, height_buf, grid, reference = make_source_and_reference(
        device, naive_pipeline, height, width, seed=0)
    buf_out_tiled = device.empty(height * width, np.float32)

    results = []
    for tile_x, tile_y in shapes:
        pipeline = device.compile(make_tiled_source(tile_x, tile_y), "gaussian_buffer_tiled")
        total = tile_x * tile_y
        if total > pipeline.max_threads_per_threadgroup or total % pipeline.thread_execution_width != 0:
            results.append({"tile": (tile_x, tile_y), "threads": total, "skipped": True})
            continue

        tg_buffers = [buf_in, buf_out_tiled, width_buf, height_buf]
        ms = gpu_median_ms(pipeline, tg_buffers, grid, (tile_x, tile_y), repeat)

        out = buf_out_tiled.contents.reshape(height, width)
        max_diff = float(np.abs(out - reference).max())

        results.append({
            "tile": (tile_x, tile_y),
            "threads": total,
            "ms": ms,
            "max_diff": max_diff,
            "skipped": False,
        })
    return results


def demonstrate_mismatch_is_wrong(device: Device, naive_pipeline) -> None:
    """Compiles the tiled kernel for a 32x32 tile, then dispatches it with a
    threadgroup shape that does *not* match -- correctness only holds when
    threadgroup= matches the compiled TILE_X/TILE_Y, and this shows what
    happens when it doesn't."""
    height, width = 257, 257
    buf_in, width_buf, height_buf, grid, reference = make_source_and_reference(
        device, naive_pipeline, height, width, seed=1)
    buf_out_tiled = device.empty(height * width, np.float32)

    pipeline = device.compile(make_tiled_source(32, 32), "gaussian_buffer_tiled")

    pipeline.run([buf_in, buf_out_tiled, width_buf, height_buf], grid,
                 wait=True, threadgroup=(32, 32))
    matched_diff = float(np.abs(buf_out_tiled.contents.reshape(height, width) - reference).max())

    pipeline.run([buf_in, buf_out_tiled, width_buf, height_buf], grid,
                 wait=True, threadgroup=(16, 16))
    mismatched_diff = float(np.abs(buf_out_tiled.contents.reshape(height, width) - reference).max())

    print("Correctness pitfall: TILE_X=TILE_Y=32 kernel, threadgroup= matched vs mismatched")
    print(f"  threadgroup=(32, 32) (matches TILE_X/TILE_Y): max|diff| vs untiled reference = {matched_diff:.6g}")
    print(f"  threadgroup=(16, 16) (does NOT match):         max|diff| vs untiled reference = {mismatched_diff:.6g}")
    print("  -> the explicit threadgroup= override isn't just a performance knob here; the\n"
          "     auto-computed default would hit this same mismatch since it has no way to\n"
          "     know the kernel's tile size is baked into its threadgroup-memory layout.\n")


def parse_sizes(spec: str | None) -> list[tuple[int, int]]:
    if not spec:
        return DEFAULT_SIZES
    sizes = []
    for part in spec.split(","):
        w, h = part.lower().split("x")
        sizes.append((int(h), int(w)))
    return sizes


def print_table(height: int, width: int, results: list[dict]) -> None:
    print(f"\n{width}x{height}")
    header = f"{'tile (w x h)':>14} {'threads':>8} {'gpu ms':>10} {'max|diff|':>11}"
    print(header)
    print("-" * len(header))
    timed = [r for r in results if not r["skipped"]]
    fastest_ms = min((r["ms"] for r in timed), default=None)
    for r in results:
        tile_str = f"{r['tile'][0]}x{r['tile'][1]}"
        if r["skipped"]:
            print(f"{tile_str:>14} {r['threads']:>8} {'skipped':>10} {'(invalid)':>11}")
            continue
        marker = "*" if r["ms"] == fastest_ms else " "
        print(f"{tile_str:>14} {r['threads']:>8} {r['ms']:>9.4f}{marker} {r['max_diff']:>11.2e}")
    print("* = fastest for this size")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=str, default=None,
                         help=f"comma-separated WxH pairs (default: {DEFAULT_SIZES})")
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    args = parser.parse_args()

    device = Device()
    naive_pipeline = device.compile(NAIVE_KERNEL_PATH.read_text(), "gaussian_buffer")

    demonstrate_mismatch_is_wrong(device, naive_pipeline)

    for height, width in parse_sizes(args.sizes):
        results = bench_size(device, naive_pipeline, height, width, DEFAULT_SHAPES, args.repeat)
        print_table(height, width, results)

        bad = [r for r in results if not r["skipped"] and r["max_diff"] > 1e-4]
        if bad:
            print(f"  WARNING: {len(bad)} shape(s) disagreed with the reference beyond tolerance: "
                  f"{[r['tile'] for r in bad]}")


if __name__ == "__main__":
    main()
