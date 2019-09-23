import pytest
import numpy as np

from orix.vector.neo_euler import Rodrigues
from orix.quaternion.rotation import Rotation


@pytest.mark.parametrize('rotation, expected', [
    (Rotation([1, 0, 0, 0]), [0, 0, 0]),
    (Rotation([0.9239, 0.2209, 0.2209, 0.2209]), [0.2391, 0.2391, 0.2391]),
])
def test_from_rotation(rotation, expected):
    rodrigues = Rodrigues.from_rotation(rotation)
    assert np.allclose(rodrigues.data, expected, atol=1e-4)


@pytest.mark.parametrize('rodrigues, expected', [
    (Rodrigues([0.2391, 0.2391, 0.2391]), np.pi/4),
])
def test_angle(rodrigues, expected):
    angle = rodrigues.angle
    assert np.allclose(angle.data, expected, atol=1e-3)
