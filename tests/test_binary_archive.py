"""Correctness for the *explicit* pipeline-caching controls added on top of
the implicit one every Device already has (see test_pipeline_persistence.py
for that one): Device(cache_path=...)/.pipeline_cache_path/.pipeline_cache_size,
and Device.binary_archive()/.compile(archive=...)/BinaryArchive.save() for a
user-managed MTL::BinaryArchive independent of a Device's own internal cache.
"""
import numpy as np
import pytest

try:
    from mtlpy import Device, shader
    HAS_METAL = True
except Exception:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal not available")


def test_default_cache_path_is_nonempty(device):
    assert device.pipeline_cache_path != ""


def test_custom_cache_path_used_instead_of_default(tmp_path):
    custom = tmp_path / "custom.metallib"
    dev = Device(cache_path=str(custom))
    assert dev.pipeline_cache_path == str(custom)

    dev.compile(shader.add_kernel("float"), "add")
    dev.flush_cache()

    assert custom.exists()


def test_cache_path_false_disables_on_disk_caching(tmp_path, monkeypatch):
    # Sanity check this doesn't fall back to the real default location --
    # redirect $HOME so a bug that ignores cache_path=False would write to
    # somewhere this test can see, not silently pass by writing to the
    # developer's actual ~/Library/Caches/mtlpy/.
    monkeypatch.setenv("HOME", str(tmp_path))
    dev = Device(cache_path=False)
    assert dev.pipeline_cache_path == ""

    dev.compile(shader.add_kernel("float"), "add")
    dev.flush_cache()  # must be a harmless no-op, not raise

    assert not (tmp_path / "Library").exists()


def test_pipeline_cache_size_dedupes_identical_source(device):
    before = device.pipeline_cache_size
    device.compile(shader.add_kernel("float"), "add")
    after_first = device.pipeline_cache_size
    assert after_first == before + 1

    device.compile(shader.add_kernel("float"), "add")  # identical source+fn
    assert device.pipeline_cache_size == after_first  # no new entry

    device.compile(shader.mul_kernel("float"), "mul")  # different source
    assert device.pipeline_cache_size == after_first + 1


def test_flush_cache_path_override_does_not_move_default_location(tmp_path, device):
    override = tmp_path / "override.metallib"
    device.compile(shader.add_kernel("float"), "add")
    device.flush_cache(str(override))

    assert override.exists()
    # The Device's own configured location is unchanged by a one-off
    # override -- a *later* plain flush_cache() still targets it.
    assert device.pipeline_cache_path != str(override)


def test_binary_archive_save_and_reload_produces_correct_pipeline(tmp_path):
    archive_path = tmp_path / "explicit.metallib"

    producer = Device()
    archive = producer.binary_archive(str(archive_path))
    mul = producer.compile(shader.mul_kernel("float"), "mul", archive=archive)
    a = producer.buffer(np.array([2.0, 3.0], dtype=np.float32))
    b = producer.buffer(np.array([4.0, 5.0], dtype=np.float32))
    out = producer.empty(2, np.float32)
    mul.run([a, b, out], 2)
    np.testing.assert_allclose(out.contents, [8.0, 15.0])

    archive.save()
    assert archive_path.exists()

    # A fresh Device + a fresh BinaryArchive opened from the saved path --
    # this only proves the archive round-trips to something Metal accepts
    # and dispatches correctly, not that it skipped recompilation (that's
    # a timing claim, which test_pipeline_persistence.py already documents
    # as too noisy/environment-dependent to assert on).
    consumer = Device()
    reloaded = consumer.binary_archive(str(archive_path))
    mul2 = consumer.compile(shader.mul_kernel("float"), "mul", archive=reloaded)
    a2 = consumer.buffer(np.array([6.0, 7.0], dtype=np.float32))
    b2 = consumer.buffer(np.array([8.0, 9.0], dtype=np.float32))
    out2 = consumer.empty(2, np.float32)
    mul2.run([a2, b2, out2], 2)
    np.testing.assert_allclose(out2.contents, [48.0, 63.0])


def test_archive_registers_pipeline_even_on_in_memory_cache_hit(tmp_path, device):
    """compile()'s in-memory dedup (see test_pipeline_cache_size_dedupes_identical_source
    above) must not skip registering into an explicitly-passed archive just
    because this exact (source, function_name) was already compiled once on
    this same Device without one."""
    device.compile(shader.add_kernel("float"), "add")  # no archive -- populates the in-memory cache

    archive_path = tmp_path / "hit.metallib"
    archive = device.binary_archive(str(archive_path))
    device.compile(shader.add_kernel("float"), "add", archive=archive)  # cache hit, archive= given
    archive.save()

    assert archive_path.exists()
    assert archive_path.stat().st_size > 0


def test_binary_archive_save_with_no_path_anywhere_raises():
    dev = Device()
    archive = dev.binary_archive()  # no path given at creation either
    with pytest.raises(RuntimeError):
        archive.save()


def test_binary_archive_accumulates_multiple_pipelines(tmp_path):
    archive_path = tmp_path / "multi.metallib"
    dev = Device()
    archive = dev.binary_archive(str(archive_path))

    dev.compile(shader.add_kernel("float"), "add", archive=archive)
    dev.compile(shader.mul_kernel("float"), "mul", archive=archive)
    archive.save()

    consumer = Device()
    reloaded = consumer.binary_archive(str(archive_path))
    add = consumer.compile(shader.add_kernel("float"), "add", archive=reloaded)
    mul = consumer.compile(shader.mul_kernel("float"), "mul", archive=reloaded)

    a = consumer.buffer(np.array([1.0, 2.0], dtype=np.float32))
    b = consumer.buffer(np.array([3.0, 4.0], dtype=np.float32))
    out_add = consumer.empty(2, np.float32)
    out_mul = consumer.empty(2, np.float32)
    add.run([a, b, out_add], 2)
    mul.run([a, b, out_mul], 2)

    np.testing.assert_allclose(out_add.contents, [4.0, 6.0])
    np.testing.assert_allclose(out_mul.contents, [3.0, 8.0])
