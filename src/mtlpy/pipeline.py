from __future__ import annotations

# Sentinel distinguishing "wait not passed" from "wait=True passed explicitly"
# -- needed because True is also wait's own default, so a plain `wait: bool =
# True` parameter can't tell the two apart. See Pipeline.run's cb/wait check.
_WAIT_UNSET = object()


class Pipeline:
    def __init__(self, _pipeline):
        self._pipeline = _pipeline  # _mtlpy.Pipeline

    def run(self, buffers: list, grid, wait=_WAIT_UNSET,
            textures: list | None = None, samplers: list | None = None,
            cb: "CommandBuffer | None" = None) -> tuple[float, float]:
        """Returns (gpu_start, gpu_end) in seconds -- pure device-side
        execution time for this dispatch (MTLCommandBuffer's GPUStartTime/
        GPUEndTime), excluding CPU-side encoding/dispatch overhead. Only
        meaningful when wait=True and cb is None; (0.0, 0.0) when wait=False,
        or whenever cb is given (see CommandBuffer -- once dispatches share a
        command buffer, only CommandBuffer.commit()'s combined timing is
        meaningful, not any individual dispatch's).

        cb batches this dispatch into an existing CommandBuffer (see
        Device.command_buffer()) instead of submitting its own command
        buffer; the CommandBuffer's own .commit(wait) controls waiting,
        once, after every dispatch you want batched together has been
        encoded into it. wait has no meaning in that case, so passing it
        explicitly alongside cb raises ValueError instead of silently
        ignoring it -- a caller who adds cb=cb to an existing wait=True (or
        wait=False) call site should have to notice wait no longer does
        anything, not have it silently stop applying."""
        if cb is not None and wait is not _WAIT_UNSET:
            raise ValueError(
                "wait has no effect when cb is given -- control waiting via "
                "CommandBuffer.commit(wait) once, after every dispatch you want "
                "batched together has been encoded, not per-dispatch"
            )
        wait = True if wait is _WAIT_UNSET else wait
        if isinstance(grid, int):
            grid = [grid, 1, 1]
        return self._pipeline.run(
            [b._buf for b in buffers],
            [t._tex for t in (textures or [])],
            [s._sampler for s in (samplers or [])],
            list(grid),
            wait,
            cb._cb if cb is not None else None,
        )

    @property
    def thread_execution_width(self) -> int:
        return self._pipeline.thread_execution_width()

    @property
    def max_threads_per_threadgroup(self) -> int:
        return self._pipeline.max_threads_per_threadgroup()


class CommandBuffer:
    """Batches multiple Pipeline.run() dispatches into one MTLCommandBuffer
    submission, instead of each dispatch paying its own command-buffer-
    create + commit + waitUntilCompleted round trip -- see
    Device.command_buffer(). Used as a context manager: dispatches encoded
    into it (via Pipeline.run(..., cb=cb)) inside the `with` block are all
    submitted together when the block exits.

        with device.command_buffer() as cb:
            pipeline1.run(bufs1, grid1, cb=cb)
            pipeline2.run(bufs2, grid2, cb=cb)
        # one submit, one wait, covering both dispatches

    If the `with` block raises, __exit__ does *not* commit -- partially
    encoded work is discarded rather than submitted, matching what you'd
    want from a failed batch (the same reasoning a database transaction
    context manager rolls back on exception instead of committing a partial
    write). That guarantee also holds without the context manager: if any
    Pipeline.run(cb=cb) call raises (e.g. an unbound-argument error), this
    CommandBuffer is marked failed at the C++ layer, and commit() -- however
    you reach it -- then raises too instead of silently submitting whatever
    was successfully encoded before the failure."""

    def __init__(self, _cb):
        self._cb        = _cb    # _mtlpy.CommandBuffer
        self._committed = False

    def __enter__(self) -> "CommandBuffer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None and not self._committed:
            self.commit()

    def commit(self, wait: bool = True) -> tuple[float, float]:
        """Ends encoding and submits every dispatch encoded into this
        CommandBuffer so far. Returns (gpu_start, gpu_end) in seconds --
        combined device-side execution time across every batched dispatch,
        not per-dispatch -- when wait=True; (0.0, 0.0) otherwise. Raises if
        called more than once (including implicitly, via the context
        manager's __exit__), or if a dispatch encoded into this
        CommandBuffer previously failed (see the class docstring)."""
        if self._committed:
            raise RuntimeError("CommandBuffer already committed")
        self._committed = True
        return self._cb.commit(wait)
