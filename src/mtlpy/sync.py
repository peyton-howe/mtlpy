from __future__ import annotations


class Queue:
    """A second (or third, ...) MTL::CommandQueue beyond the one every
    Device already owns internally -- see Device.queue(). Only meaningful
    passed to Device.command_buffer(queue=...): it lets that batch of
    Pipeline.run(..., cb=cb) dispatches run on an independent GPU
    scheduling stream, concurrently with whatever else the Device's own
    default queue (or another Queue) is doing, synchronized only where an
    Event or SharedEvent explicitly says so (see CommandBuffer.wait_for_event/
    .signal_event). A Pipeline's *self-contained* dispatch (Pipeline.run()
    with no cb=) always targets the Device's own default queue regardless of
    what Queues exist -- there's no way to redirect that path, only the
    batched (cb=) path is queue-selectable."""

    def __init__(self, _queue, device):
        self._queue = _queue  # _mtlpy.Queue
        self._device = device


class Fence:
    """A lightweight producer/consumer ordering primitive between two
    compute dispatches on the *same* MTL::CommandQueue (either two Pipeline.run()
    calls batched into one CommandBuffer, or two separate ones/CommandBuffers
    submitted to the same queue) -- see Pipeline.run's wait_fences/signal_fences.

    Every Buffer/Texture this library hands out uses Metal's automatic
    resource-hazard tracking (none are created with hazardTrackingModeUntracked),
    which already orders a dispatch that reads a buffer after an earlier one
    that wrote it -- on the same queue, that ordering happens for free,
    without needing a Fence at all. Fence exists as an explicit, lower-level
    tool for orderings that aren't implied by resource usage alone. For
    ordering dispatches across *different* queues (see Queue), use Event or
    SharedEvent instead -- Fence's guarantee is scoped to a single queue."""

    def __init__(self, _fence, device):
        self._fence = _fence  # _mtlpy.Fence
        self._device = device


class Event:
    """A GPU-side-only synchronization primitive for ordering work across
    separate CommandBuffers -- including ones submitted to *different*
    Queues, which is the one ordering guarantee a shared queue's own
    commit-order semantics don't give you for free. Signaled/waited on via
    CommandBuffer.signal_event()/.wait_for_event(), encoded as part of a
    batch rather than called directly on this object -- there's no
    CPU-visible state at all here (see SharedEvent for that), so nothing on
    Event itself is callable from Python beyond construction.

    Typical producer/consumer pattern across two queues:

        q1, q2 = device.queue(), device.queue()
        event = device.event()

        with device.command_buffer(queue=q1) as cb1:
            producer.run(bufs, grid, cb=cb1)
            cb1.signal_event(event, 1)

        with device.command_buffer(queue=q2) as cb2:
            cb2.wait_for_event(event, 1)
            consumer.run(bufs, grid, cb=cb2)
    """

    def __init__(self, _event, device):
        self._event = _event  # _mtlpy.Event
        self._device = device


class SharedEventHandle:
    """An opaque, exportable reference to a SharedEvent
    (SharedEvent.export_handle()) that another process can import via
    Device.import_shared_event() to synchronize with that same event across
    a process boundary. Actually transporting this handle between processes
    (e.g. over an XPC connection -- MTLSharedEventHandle conforms to
    NSSecureCoding, so NSXPCConnection knows how to encode it natively) is
    the caller's responsibility; mtlpy only provides the create/export/import
    primitives here, not an IPC channel of its own."""

    def __init__(self, _handle):
        self._handle = _handle  # _mtlpy.SharedEventHandle


class SharedEvent(Event):
    """Like Event, but adds a CPU-visible uint64 "signaled value" that can be
    set/read directly from Python (signal()/.signaled_value) and blocked on
    from Python (wait()) -- the mechanism for CPU<->GPU handoff:

        event = device.shared_event()

        # GPU signals, CPU waits:
        with device.command_buffer() as cb:
            producer.run(bufs, grid, cb=cb)
            cb.signal_event(event, 1)
        event.wait(1)  # blocks until the GPU-side signal above fires

        # CPU signals, GPU waits:
        with device.command_buffer() as cb:
            cb.wait_for_event(event, 1)
            consumer.run(bufs, grid, cb=cb)
        cb.commit(wait=False)
        event.signal(1)  # unblocks the GPU-side wait above

    Also exportable via export_handle() for cross-process use -- see
    SharedEventHandle."""

    def signal(self, value: int) -> None:
        """Sets this event's signaled value directly from the CPU -- unblocks
        any command buffer whose encoded wait_for_event(self, value) (or
        lower) is waiting, and any Python thread blocked in wait(value) (or
        lower)."""
        self._event.signal(value)

    @property
    def signaled_value(self) -> int:
        """This event's current signaled value, read directly from the CPU
        (no waiting -- see wait() to block until it reaches a target value)."""
        return self._event.signaled_value

    def wait(self, value: int, timeout_ms: int = 5000) -> bool:
        """Blocks the calling Python thread until signaled_value reaches at
        least value, or timeout_ms elapses. Returns False on timeout, True
        once satisfied. Releases the GIL for the whole call, same as
        CommandBuffer.commit(wait=True) -- other Python threads keep running
        while this one blocks."""
        return self._event.wait(value, timeout_ms)

    def export_handle(self) -> SharedEventHandle:
        """A SharedEventHandle another process can import via
        Device.import_shared_event() to synchronize with this same event --
        see SharedEventHandle's docstring for how the handle itself actually
        needs to reach that other process."""
        return SharedEventHandle(self._event.new_shared_event_handle())
