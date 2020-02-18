import pytest
import numpy

from orix.quaternion.symmetry import *
from orix.quaternion.orientation import Orientation
from orix.quaternion.orientation_region import (
    _get_large_cell_normals,
    get_proper_groups,
    OrientationRegion,
)
from orix.quaternion.symmetry import get_distinguished_points


@pytest.mark.parametrize(
    "s1, s2, expected",
    [
        (C2, C1, [[0, 0, 0, 1], [0, 0, 0, -1]]),
        (
            C3,
            C1,
            [
                [0.5, 0, 0, 0.866],
                [-0.5, 0, 0, -0.866],
                [-0.5, 0, 0, 0.866],
                [0.5, 0, 0, -0.866],
            ],
        ),
        (
            D3,
            C3,
            [
                [0.5, 0.0, 0.0, 0.866],
                [-0.5, 0.0, 0.0, -0.866],
                [-0.5, 0.0, 0.0, 0.866],
                [0.5, -0.0, -0.0, -0.866],
                [0.0, 1.0, 0.0, 0.0],
                [-0.0, -1.0, -0.0, -0.0],
                [0.0, 0.5, 0.866, 0.0],
                [-0.0, -0.5, -0.866, 0.0],
                [0.0, -0.5, 0.866, 0.0],
                [0.0, 0.5, -0.866, 0.0],
            ],
        ),
    ],
)
def test_get_distinguished_points(s1, s2, expected):
    dp = get_distinguished_points(s1, s2)
    assert np.allclose(dp.data, expected, atol=1e-3)


@pytest.mark.parametrize(
    "s1, s2, expected",
    [
        (C2, C1, [[0.5 ** 0.5, 0, 0, -(0.5 ** 0.5)], [0.5 ** 0.5, 0, 0, 0.5 ** 0.5]]),
        (C6, C1, [[0.258819, 0, 0, -0.965926,], [0.258819, 0, 0, 0.965926,]]),
        (C3, C3, [[0.5, 0, 0, -0.866], [0.5, 0, 0, 0.866]]),
        (
            D2,
            C1,
            [
                [0.5 ** 0.5, -(0.5 ** 0.5), 0, 0,],
                [0.5 ** 0.5, 0, -(0.5 ** 0.5), 0,],
                [0.5 ** 0.5, 0, 0, -(0.5 ** 0.5),],
                [0.5 ** 0.5, 0, 0, 0.5 ** 0.5,],
                [0.5 ** 0.5, 0, 0.5 ** 0.5, 0,],
                [0.5 ** 0.5, 0.5 ** 0.5, 0, 0,],
            ],
        ),
        (
            D3,
            C1,
            [
                [0.707107, -0.707107, 0, 0,],
                [0.707107, -0.353553, -0.612372, 0,],
                [0.707107, -0.353553, 0.612372, 0,],
                [0.5, 0, 0, -0.866025,],
                [0.5, 0, 0, 0.866025,],
                [0.707107, 0.353553, -0.612372, 0,],
                [0.707107, 0.353553, 0.612372, 0,],
                [0.707107, 0.707107, 0, 0,],
            ],
        ),
        (
            D6,
            C1,
            [
                [0.707107, -0.707107, 0, 0,],
                [0.707107, -0.612372, -0.353553, 0,],
                [0.707107, -0.612372, 0.353553, 0,],
                [0.707107, -0.353553, -0.612372, 0,],
                [0.707107, -0.353553, 0.612372, 0,],
                [0.707107, 0, -0.707107, 0,],
                [0.258819, 0, 0, -0.965926,],
                [0.258819, 0, 0, 0.965926,],
                [0.707107, 0, 0.707107, 0,],
                [0.707107, 0.353553, -0.612372, 0,],
                [0.707107, 0.353553, 0.612372, 0,],
                [0.707107, 0.612372, -0.353553, 0,],
                [0.707107, 0.612372, 0.353553, 0,],
                [0.707107, 0.707107, 0, 0,],
            ],
        ),
    ],
)
def test_get_large_cell_normals(s1, s2, expected):
    n = _get_large_cell_normals(s1, s2)
    print(n)
    assert np.allclose(n.data, expected, atol=1e-3)


def test_coverage_on_faces():
    o = OrientationRegion(Orientation([1, 1, 1, 1]))
    f = o.faces()
    return None


@pytest.mark.parametrize(
    "Gl,Gr",
    [
        (C1, Ci),
        (Ci, C1),
        (C1, Csz),
        (Csz, C1),
        (Ci, Csz),
        (Csz, Ci),
        (C1, C1),
        (Ci, Ci),
    ],
)
def test_get_proper_point_groups(Gl, Gr):
    get_proper_groups(Gl, Gr)
    return None


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_get_proper_point_group_not_implemented():
    """ Double inversion case not yet implemented """
    get_proper_groups(Csz, Csz)
