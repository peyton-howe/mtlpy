from __future__ import annotations


class BinaryArchive:
    """An explicit, user-managed precompiled-pipeline cache (MTL::BinaryArchive)
    -- independent of the implicit one every Device already maintains for
    you (see Device.flush_cache()/.pipeline_cache_path). Useful when that
    default, single, per-user-home-directory cache doesn't fit: a custom
    location, sharing one archive across several Devices in the same
    process, or building an archive meant to ship as a read-only asset with
    an app (compile everything once during a build step, .save() the
    result, then Device.binary_archive(path=...) to load it back at runtime
    for fast first-launch pipeline creation with no shader recompilation).

    Pass to Device.compile(..., archive=...) to have that specific compile
    additionally register into this archive, on top of whatever the
    Device's own internal cache already does -- works whether or not the
    compile actually recompiles from source: a cache hit against the
    Device's own internal cache still registers the already-compiled
    pipeline here, no extra shader compilation needed either way."""

    def __init__(self, _archive, device):
        self._archive = _archive  # _mtlpy.BinaryArchive
        self._device = device

    def save(self, path: str | None = None) -> None:
        """Serializes this archive to disk at path (defaults to the path
        this archive was created/opened with via Device.binary_archive() --
        raises if that was never given one either)."""
        self._archive.save(str(path) if path is not None else "")
