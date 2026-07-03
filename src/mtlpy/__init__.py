import sys

if sys.platform != "darwin":
    raise ImportError("mtlpy requires macOS with Apple Metal support")

from .device import Device
from .buffer import Buffer
from . import operators

__version__ = "0.1.0"
__all__ = ["Device", "Buffer", "operators"]
