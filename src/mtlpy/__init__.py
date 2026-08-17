import sys
from importlib.metadata import version as _version

if sys.platform != "darwin":
    raise ImportError("mtlpy requires macOS with Apple Metal support")

from .device import Device, list_devices
from .binary_archive import BinaryArchive
from .buffer import Buffer
from .capture import Capture, CaptureScope
from .heap import Heap
from .sync import Event, Fence, Queue, SharedEvent, SharedEventHandle
from .texture import Sampler, Texture
from .utils import StorageMode
from . import operators

# Read from the installed package's own metadata (which scikit-build-core
# generates from pyproject.toml's [project].version at build time) instead
# of a hardcoded string.
__version__ = _version("mtlpy")
__all__ = [
    "Device", "Buffer", "Heap", "Texture", "Sampler", "StorageMode", "operators",
    "list_devices", "Event", "SharedEvent", "SharedEventHandle", "Fence", "Queue",
    "BinaryArchive", "Capture", "CaptureScope",
]
