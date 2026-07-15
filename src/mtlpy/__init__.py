import sys

if sys.platform != "darwin":
    raise ImportError("mtlpy requires macOS with Apple Metal support")

from .device import Device, list_devices
from .buffer import Buffer
from .texture import Sampler, Texture
from . import operators

__version__ = "0.1.0"
__all__ = ["Device", "Buffer", "Texture", "Sampler", "operators", "list_devices"]
