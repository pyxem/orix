import pytest
import numpy

from texpy.quaternion.symmetry import *

from texpy.quaternion.orientation_region import _get_large_cell_normals
from texpy.quaternion.symmetry import get_distinguished_points


@pytest.mark.parametrize('s1, s2, expected', [
    (C2, C1, [[0, 0, 0, 1], [0, 0, 0, -1]]),
    (C3, C1, [
        [0.5, 0, 0, 0.866],
        [-0.5, 0, 0, -0.866],
        [-0.5, 0, 0, 0.866],
        [0.5, 0, 0, -0.866],
    ]),
    (D3, C3, [
        [ 0.5,    0. ,    0.   ,  0.866],
        [-0.5,    0. ,    0.   , -0.866],
        [-0.5,    0. ,    0.   ,  0.866],
        [ 0.5,   -0. ,   -0.   , -0.866],
        [ 0. ,    1. ,    0.   ,  0.   ],
        [-0. ,   -1. ,   -0.   , -0.   ],
        [ 0. ,    0.5,    0.866,  0.   ],
        [-0. ,   -0.5,   -0.866,  0.   ],
        [ 0. ,   -0.5,    0.866,  0.   ],
        [ 0. ,    0.5,   -0.866,  0.   ],
    ])
])
def test_get_distinguished_points(s1, s2, expected):
    dp = get_distinguished_points(s1, s2)
    assert np.allclose(dp.data, expected, atol=1e-3)


@pytest.mark.parametrize('s1, s2, expected', [
    (C2, C1, [[0.5**0.5, 0, 0, 0.5**0.5], [0.5**0.5, 0, 0, -0.5**0.5]]),
    (C3, C3, [[0.866, 0, 0, 0.5], [0.866, 0, 0, -0.5], [0.5, 0, 0, -0.866], [0.5, 0, 0, 0.866]])
])
def test_get_large_cell_normals(s1, s2, expected):
    n = _get_large_cell_normals(s1, s2)
    assert np.allclose(n.data, expected, atol=1e-3)


