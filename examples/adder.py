"""Minimal example: add two float arrays on the GPU."""
import numpy as np
import mtlpy

device = mtlpy.Device()

a = device.buffer(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
b = device.buffer(np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32))

c = a + b

print("a:    ", a.contents)
print("b:    ", b.contents)
print("a + b:", c.contents)
