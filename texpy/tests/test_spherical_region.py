import numpy as np
import pytest

from texpy.quaternion.symmetry import Symmetry
from texpy.vector.spherical_region import SphericalRegion
from texpy.vector.vector3d import Vector3d


@pytest.fixture
def spherical_region(request):
    sr = SphericalRegion()
    sr.normals = request.param
    return sr

@pytest.fixture
def vector(request):
    v = Vector3d(request.param)
    return v


@pytest.mark.parametrize('spherical_region, vector', [
    ([[0, 0, 1], [0, 1, 0], [1, 0, 0]], [0.1, 0.1, 0.1]),
    ([[0, 0, 1], [0, 1, 0], [1, 0, 0]], [1, 1, 1]),
    pytest.param([[0, 0, 1], [0, 1, 0], [1, 0, 0]], [0.1, -0.1, 0.1],
                 marks=pytest.mark.xfail),
    pytest.param([[0, 0, 1], [0, 1, 0], [1, 0, 0]], [[0.1, -0.1, 0.1], [1, 1, 1]],
                 marks=pytest.mark.xfail),
], indirect=['spherical_region', 'vector'])
def test_is_inside(spherical_region, vector):
    assert np.all(spherical_region.contains(vector))


@pytest.mark.parametrize('symbol, normals', [
    ('2', np.array([
        [-1, 0, 0]
    ])),
    ('m', np.array([
        [0, 0, 1],
    ])),
    ('2/m', np.array([
        [0, 0, 1],
        [-1, 0, 0],
    ])),
    # ('222', np.array([
    #     [0, 1, 0],
    #     [1, 0, 0],
    # ])),
    ('4mm', np.array([
        [-1, 0, 0],
        [0.707107, 0.707107, 0],
    ])),
    ('mm2', np.array([
        [-1, 0, 0],
        [ 0, 1, 0],
    ])),
    ('-4', np.array([
        [0, 0, 1],
        [-1, 0, 0],
    ])),
    ('32', np.array([
        [  -1,        0,        0],
        [-0.5, 0.866025,        0],
        [   0,        0,        1],
    ])),
    # ('3m', np.array([
    #     [-1, 0, 0],
    #     [0.5, 0.866025, 0],
    # ]))
    ('-42m', np.array([
        [-0.707107, 0.707107, 0],
        [ 0.707107, 0.707107, 0],
        [0, 0, 1],
    ])),
    # ('m-3m', np.array([
    #     [0, 1, 0],
    #     [0.707107, -0.707107, 0],
    #     [-0.707107, 0, 0.707107],
    # ]))
])
def test_from_symmetry(symbol, normals):
    s = Symmetry.from_symbol(symbol)
    sr = SphericalRegion.from_symmetry(s)
    print(sr.normals.data.round(4))
    assert np.allclose(sr.normals.data, normals, atol=1e-4)