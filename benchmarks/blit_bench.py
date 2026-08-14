#!/usr/bin/env python3
"""Timing for CPU<->Texture upload/download, across the methods available:

  upload:   .upload()            DEFAULT -- blits through an internal
                                  Buffer sub-allocated from a Device-owned,
                                  auto-growing Heap (Device._staging_buffer())
            _raw_replaceregion    the old mechanism .upload() used to use
                                  (MTL::Texture::replaceRegion), exercised
                                  directly via the low-level _tex.upload()
                                  binding, for a before/after comparison
            .upload_from_buffer()  blits a Buffer you supply and manage
                                  yourself -- skips the internal staging
                                  Buffer/Heap indirection entirely

  download: .download()          DEFAULT -- blits into an internal Heap-
                                  backed Buffer, then copies out into an
                                  ordinary independent array (see its
                                  docstring for why the copy-out, not a
                                  zero-copy view)
            _raw_getbytes         the old mechanism (MTL::Texture::getBytes),
                                  exercised via the low-level _tex.download()
                                  binding, for a before/after comparison
            .download_to_buffer()  blits into a Buffer you supply and
                                  manage yourself

Each is measured two ways:
  - one-shot: nothing reused across calls -- what .upload()/.download()
    always look like from the outside (their internal staging Buffer is
    never exposed to reuse).
  - hot loop: upload_from_buffer()/download_to_buffer() given a single
    Buffer YOU allocate once and reuse across every call (new data written
    into it each time for upload) -- skips the internal Heap indirection
    .upload()/.download() pay for on every call, so it's the fastest
    option once you're willing to manage the Buffer yourself.

Usage:
    python benchmarks/blit_bench.py
    python benchmarks/blit_bench.py --sizes 480x640,1080x1920 --repeat 50

Same wall-clock, warm-dispatch methodology as benchmarks/bench.py (see its
module docstring) -- duplicated rather than imported, since each script
here is meant to be standalone.
"""
from __future__ import annotations

import argparse
import itertools
import statistics
import time

import numpy as np

from mtlpy import Device

DEFAULT_SIZES = [(480, 640), (1080, 1920), (2160, 3840)]  # (height, width)


def _data_pool(height: int, width: int, n: int = 8) -> list[np.ndarray]:
    """n distinct arrays, precomputed outside any timed loop, to cycle
    through so a hot-loop benchmark delivers genuinely new data each
    iteration without the cost of generating a fresh random array every
    iteration polluting the measurement."""
    rng = np.random.default_rng(0)
    return [rng.random((height, width), dtype=np.float32) for _ in range(n)]


def warmup_until_stable(
    dispatch, max_iters: int = 100, window: int = 8, tol: float = 0.08, min_iters: int = 8
) -> None:
    times = []
    for i in range(max_iters):
        t0 = time.perf_counter()
        dispatch()
        times.append(time.perf_counter() - t0)
        if i + 1 >= min_iters and i + 1 >= window:
            recent = times[-window:]
            spread = (max(recent) - min(recent)) / statistics.median(recent)
            if spread < tol:
                break


def timed(dispatch, repeat: int, warmup: int = 8) -> float:
    """Median wall-clock milliseconds per call, after warmup."""
    for _ in range(warmup):
        dispatch()
    warmup_until_stable(dispatch)
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        dispatch()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def _raw_replaceregion(tex, arr: np.ndarray) -> None:
    """Exercises the old mechanism .upload() used before this session --
    MTL::Texture::replaceRegion(), via the low-level _tex.upload() binding
    directly (bypassing the Python Texture.upload(), which is now
    blit-based) -- kept only so this benchmark can still show a before/
    after comparison."""
    bytes_per_row, bytes_per_image = tex._bytes_per_row_and_image()
    tex._tex.upload(arr, bytes_per_row, bytes_per_image)


def _raw_getbytes(tex) -> np.ndarray:
    """Exercises the old mechanism .download() used before this session --
    MTL::Texture::getBytes(), via the low-level _tex.download() binding
    directly (bypassing the Python Texture.download(), which is now
    blit-based) -- kept only so this benchmark can still show a before/
    after comparison."""
    bytes_per_row, bytes_per_image = tex._bytes_per_row_and_image()
    nbytes = tex.width * tex.height * tex.channels * tex.dtype.itemsize
    raw = tex._tex.download(nbytes, bytes_per_row, bytes_per_image)
    return np.frombuffer(raw, dtype=tex.dtype).reshape(tex.shape)


# ---------------------------------------------------------------------------
# Upload: one-shot
# ---------------------------------------------------------------------------

def upload_oneshot_raw_replaceregion(device: Device, height: int, width: int, repeat: int) -> float:
    tex = device.empty_texture((height, width), "r32Float")
    data = np.zeros((height, width), dtype=np.float32)
    return timed(lambda: _raw_replaceregion(tex, data), repeat)


def upload_oneshot_default(device: Device, height: int, width: int, repeat: int) -> float:
    tex = device.empty_texture((height, width), "r32Float")
    data = np.zeros((height, width), dtype=np.float32)
    return timed(lambda: tex.upload(data), repeat)


# ---------------------------------------------------------------------------
# Upload: hot loop -- .upload() (repeated) vs a Buffer YOU reuse
# ---------------------------------------------------------------------------

def upload_hotloop_default(device: Device, height: int, width: int, repeat: int) -> float:
    tex = device.empty_texture((height, width), "r32Float")
    pool = _data_pool(height, width)
    counter = itertools.count()
    return timed(lambda: tex.upload(pool[next(counter) % len(pool)]), repeat)


def upload_hotloop_own_buffer(device: Device, height: int, width: int, repeat: int) -> float:
    tex = device.empty_texture((height, width), "r32Float")
    staging = device.empty(height * width, np.float32)  # allocated ONCE, by the caller
    pool = [p.reshape(-1) for p in _data_pool(height, width)]
    counter = itertools.count()

    def dispatch():
        staging.contents[:] = pool[next(counter) % len(pool)]
        tex.upload_from_buffer(staging)

    return timed(dispatch, repeat)


# ---------------------------------------------------------------------------
# Download: one-shot
# ---------------------------------------------------------------------------

def download_oneshot_raw_getbytes(device: Device, height: int, width: int, repeat: int) -> float:
    tex = device.empty_texture((height, width), "r32Float")
    tex.upload(np.zeros((height, width), dtype=np.float32))
    return timed(lambda: _raw_getbytes(tex), repeat)


def download_oneshot_default(device: Device, height: int, width: int, repeat: int) -> float:
    tex = device.empty_texture((height, width), "r32Float")
    tex.upload(np.zeros((height, width), dtype=np.float32))
    return timed(lambda: tex.download(), repeat)


# ---------------------------------------------------------------------------
# Download: hot loop -- .download() (repeated) vs a Buffer YOU reuse
# ---------------------------------------------------------------------------

def download_hotloop_default(device: Device, height: int, width: int, repeat: int) -> float:
    tex = device.empty_texture((height, width), "r32Float")
    tex.upload(np.zeros((height, width), dtype=np.float32))
    return timed(lambda: tex.download(), repeat)


def download_hotloop_own_buffer(device: Device, height: int, width: int, repeat: int) -> float:
    tex = device.empty_texture((height, width), "r32Float")
    tex.upload(np.zeros((height, width), dtype=np.float32))
    out = device.empty(height * width, np.float32)  # allocated ONCE, by the caller

    def dispatch():
        tex.download_to_buffer(out)
        out.numpy()

    return timed(dispatch, repeat)


def print_table(title: str, sizes, rows, columns) -> None:
    print(title)
    header = f"{'size':>12} " + " ".join(f"{c:>20}" for c in columns) + "  fastest"
    print(header)
    print("-" * len(header))
    for (height, width), values in zip(sizes, rows):
        best_i = min(range(len(values)), key=lambda i: values[i])
        cells = " ".join(f"{v:>20.4f}" for v in values)
        print(f"{f'{height}x{width}':>12} {cells}  {columns[best_i]}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=str, default=None,
                         help=f"comma-separated HxW pairs (default: {DEFAULT_SIZES})")
    parser.add_argument("--repeat", type=int, default=30, help="samples per (op, size) for warm timing")
    args = parser.parse_args()

    if args.sizes:
        sizes = []
        for pair in args.sizes.split(","):
            h, w = pair.lower().split("x")
            sizes.append((int(h), int(w)))
    else:
        sizes = DEFAULT_SIZES

    device = Device()

    columns = ["replaceRegion (old, raw)", "upload()"]
    rows = [[upload_oneshot_raw_replaceregion(device, h, w, args.repeat),
             upload_oneshot_default(device, h, w, args.repeat)] for h, w in sizes]
    print_table("UPLOAD -- one-shot: old raw mechanism vs default", sizes, rows, columns)

    columns = ["upload() (repeated)", "upload_from_buffer() (own reused Buffer)"]
    rows = [[upload_hotloop_default(device, h, w, args.repeat),
             upload_hotloop_own_buffer(device, h, w, args.repeat)] for h, w in sizes]
    print_table("UPLOAD -- hot loop: default vs managing your own reused Buffer", sizes, rows, columns)

    columns = ["getBytes (old, raw)", "download()"]
    rows = [[download_oneshot_raw_getbytes(device, h, w, args.repeat),
             download_oneshot_default(device, h, w, args.repeat)] for h, w in sizes]
    print_table("DOWNLOAD -- one-shot: old raw mechanism vs default", sizes, rows, columns)

    columns = ["download() (repeated)", "download_to_buffer() (own reused Buffer)"]
    rows = [[download_hotloop_default(device, h, w, args.repeat),
             download_hotloop_own_buffer(device, h, w, args.repeat)] for h, w in sizes]
    print_table("DOWNLOAD -- hot loop: default vs managing your own reused Buffer", sizes, rows, columns)


if __name__ == "__main__":
    main()
