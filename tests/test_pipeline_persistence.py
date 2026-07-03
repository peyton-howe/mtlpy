"""Cross-process regression test for PipelineCache's on-disk binary archive
(csrc/pipeline_cache.cpp). This has to spawn separate Python processes -- an
in-process pytest fixture reuses the same Device/PipelineCache and would
never exercise loading a pipeline back out of the archive on disk.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import mtlpy  # noqa: F401
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")

ARCHIVE_PATH = Path.home() / "Library" / "Caches" / "mtlpy" / "pipelines.metallib"

_WORKER = """
import json, time
import numpy as np
from mtlpy import Device

dev = Device()
a = dev.buffer(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
b = dev.buffer(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))

t0 = time.perf_counter()
c = a + b
elapsed = time.perf_counter() - t0

print(json.dumps({"result": c.contents.tolist(), "elapsed": elapsed}))
"""


def _run_worker():
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_archive_persists_across_process_launches():
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()

    cold = _run_worker()
    np.testing.assert_allclose(cold["result"], [11.0, 22.0, 33.0, 44.0])

    assert ARCHIVE_PATH.exists(), (
        f"PipelineCache did not write a binary archive to {ARCHIVE_PATH} "
        "on process exit -- check PipelineCache::~PipelineCache()"
    )

    warm = _run_worker()
    np.testing.assert_allclose(warm["result"], [11.0, 22.0, 33.0, 44.0])

    # Timing is noisy and environment-dependent (process startup dominates),
    # so this isn't a hard assertion -- just print it for a human to eyeball.
    # Use benchmarks/bench.py for a repeatable first-dispatch measurement.
    print(
        f"\ncold-process add(): {cold['elapsed'] * 1e3:.3f} ms   "
        f"warm-process add(): {warm['elapsed'] * 1e3:.3f} ms"
    )
