import pytest
import numpy as np

from texpy.point_group import PointGroup
from texpy.vector.vector3d import Vector3d

symbols = ['2', 'm-3']

rotations = [
    np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]),
    np.array([
        [1, 0, 0, 0],
        [0.5, 0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5, 0.5],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ])
]

improper = [
    [0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
]

@pytest.fixture
def a():
    return Vector3d.xvector()

@pytest.fixture
def b():
    return Vector3d.yvector()

@pytest.fixture
def c():
    return Vector3d.zvector()


@pytest.mark.parametrize('symbol, rotation, improper', zip(symbols, rotations, improper))
def test_rotations(symbol, rotation, improper, a, b, c):
    rot = PointGroup(symbol).rotations(a, b, c)
    data = np.concatenate([r.data for r in rot])
    i = np.concatenate([r.improper for r in rot])
    assert np.allclose(data, rotation)
    assert np.allclose(i, improper)