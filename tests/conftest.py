import pytest

try:
    from mtlpy import Device
    HAS_METAL = True
except Exception:
    HAS_METAL = False


@pytest.fixture
def device():
    return Device()
