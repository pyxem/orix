import numpy as np
import pytest

from tests.test_symmetry_data import data
from texpy.point_group import PointGroup
from texpy.quaternion import Quaternion
from texpy.quaternion.symmetry_legacy import Symmetry
from texpy.vector import Vector3d


@pytest.fixture
def point_group(request):
    return PointGroup(request.param)


@pytest.fixture
def quaternion(request):
    return Quaternion(request.param)


@pytest.fixture
def symmetry(request):
    return Symmetry.from_symbol(request.param)

@pytest.mark.parametrize('point_group, quaternion, improper',
                         data,
indirect=['point_group', 'quaternion'])
def test_rotations_from_point_group(point_group, quaternion, improper):
    rotations = Symmetry.rotations_from_point_group(point_group)
    print(rotations)
    assert np.allclose(rotations.data, quaternion.data, atol=1e-4)
    assert np.allclose(rotations.improper, improper)



@pytest.mark.parametrize('symbol_1, symbol_2, disjoint', [
    ('m-3', '2', np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ])),
    ('m-3', '2/m', np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ])),
    ('4', '432', np.array([
        [        1,         0,         0,         0],
        [ 0.707107,         0,         0,  0.707107],
        [        0,         0,         0,         1],
        [-0.707107,         0,         0,  0.707107],
    ])),
    ('4/m', 'm-3m', np.array([
        [        1,         0,         0,         0],
        [ 0.707107,         0,         0,  0.707107],
        [        0,         0,         0,         1],
        [-0.707107,         0,         0,  0.707107],
        [        1,         0,         0,         0],
        [ 0.707107,         0,         0,  0.707107],
        [        0,         0,         0,         1],
        [-0.707107,         0,         0,  0.707107],
    ])),
    ('4mm', 'm-3m', np.array([
        [        1,         0,         0,         0],
        [        0,         1,         0,         0],
        [ 0.707107,         0,         0,  0.707107],
        [        0,  0.707107, -0.707107,         0],
        [        0,         0,         0,         1],
        [        0,         0,        -1,         0],
        [-0.707107,         0,         0,  0.707107],
        [        0, -0.707107, -0.707107,         0],
    ])),
])
def test_disjoint(symbol_1, symbol_2, disjoint):
    s1, s2 = Symmetry.from_symbol(symbol_1), Symmetry.from_symbol(symbol_2)
    dj = Symmetry.disjoint(s1, s2)
    assert dj.data.shape == disjoint.shape
    assert np.allclose(dj.data, disjoint)


@pytest.mark.parametrize('symbol, single, expected', [
    ('222', (0.881, 0.665, 0.123, 0.517), np.array([
        [ 0.881,  0.665,  0.123,  0.517],
        [-0.665,  0.881, -0.517,  0.123],
        [-0.517, -0.123,  0.665,  0.881],
        [ 0.123, -0.517, -0.881,  0.665],
    ])),
    ('4mm', ((1, 0, 0.5, 0), (3, 1, -1, -2),), np.array([[
        [       1,         0,       0.5,         0],
        [       3,         1,        -1,        -2],
        [       0,         1,         0,       0.5],
        [      -1,         3,         2,        -1],
        [0.707107, -0.353553,  0.353553,  0.707107],
        [ 3.53553,   1.41421,         0,  0.707107],
        [0.353553,  0.707107, -0.707107,  0.353553],
        [-1.41421,   3.53553, -0.707107,         0],
    ],
    [
        [        0,      -0.5,         0,         1],
        [        2,         1,         1,         3],
        [      0.5,         0,        -1,         0],
        [       -1,         2,        -3,         1],
        [-0.707107, -0.353553, -0.353553,  0.707107],
        [-0.707107,         0,   1.41421,   3.53553],
        [ 0.353553, -0.707107, -0.707107, -0.353553],
        [        0, -0.707107,  -3.53553,   1.41421],
    ]]))
])
def test_symmetrise_quaternion(symbol, single, expected):
    s = Symmetry.from_symbol(symbol)
    print(s.to_rotation())
    q = Quaternion(single)
    q_related = s.symmetrise(q)
    assert q_related.shape == q.shape + s.shape
    assert np.allclose(q_related.data, expected, atol=1e-3)


@pytest.mark.parametrize('symbol, single, expected', [
    ('222', (1, 1, 0), np.array([
        [1, 1, 0],
        [1, -1, 0],
        [-1, -1, 0],
        [-1, 1, 0],
    ])),
    ('4/m', (0.7, 0.6, 0.5), np.array([
        [ 0.7,  0.6,  0.5],
        [-0.6,  0.7,  0.5],
        [-0.7, -0.6,  0.5],
        [ 0.6, -0.7,  0.5],
        [-0.7, -0.6, -0.5],
        [ 0.6, -0.7, -0.5],
        [ 0.7,  0.6, -0.5],
        [-0.6,  0.7, -0.5],
    ]))
])
def test_symmetrise_vector(symbol, single, expected):
    s = Symmetry.from_symbol(symbol)
    v = Vector3d(single)
    v_related = s.symmetrise(v)
    assert v_related.shape == v.shape + s.shape
    assert np.allclose(v_related.data, expected, atol=1e-3)
