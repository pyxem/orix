import pytest
import numpy as np

from texpy.quaternion.symmetry import *
from texpy.vector import Vector3d


@pytest.fixture(params=[(1, 2, 3)])
def vector(request):
    return Vector3d(request.param)

@pytest.mark.parametrize('symmetry, vector, expected', [
    (Ci, (1, 2, 3), [(1, 2, 3), (-1, -2, -3)]),
    (Csx, (1, 2, 3), [(1, 2, 3), (-1, 2, 3)]),
    (Csy, (1, 2, 3), [(1, 2, 3), (1, -2, 3)]),
    (Csz, (1, 2, 3), [(1, 2, 3), (1, 2, -3)]),
    (C2, (1, 2, 3), [(1, 2, 3), (-1, -2, 3)]),
    (C2v, (1, 2, 3), [
        (1, 2, 3),
        (-1, -2, 3),
        (1, -2, 3),
        (-1, 2, 3),
    ]),
    (C4v, (1, 2, 3), [
        (1, 2, 3),
        (-2, 1, 3),
        (-1, -2, 3),
        (2, -1, 3),
        (-1, 2, 3),
        (2, 1, 3),
        (-2, -1, 3),
        (1, -2, 3)
    ]),
    (D4, (1, 2, 3), [
        (1, 2, 3),
        (-2, 1, 3),
        (-1, -2, 3),
        (2, -1, 3),
        (-1, 2, -3),
        (2, 1, -3),
        (-2, -1, -3),
        (1, -2, -3)
    ])
], indirect=['vector'])
def test_symmetry(symmetry, vector, expected):
    vector_calculated = [
        tuple(v) for v in np.int32(symmetry.outer(vector).unique().data)
    ]
    print('Expected\n', expected)
    print('Calculated\n', vector_calculated)
    print(symmetry.improper)
    assert set(vector_calculated) == set(expected)