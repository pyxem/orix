# Copyright 2018-2024 the orix developers
#
# This file is part of orix.
#
# orix is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# orix is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix.  If not, see <http://www.gnu.org/licenses/>.

import pytest

from orix.quaternion import Orientation, OrientationRegion
from orix.quaternion.orientation_region import (
    _get_large_cell_normals,
    get_proper_groups,
)
from orix.quaternion.symmetry import *


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
        (C2, C1, [[0.5**0.5, 0, 0, -(0.5**0.5)], [0.5**0.5, 0, 0, 0.5**0.5]]),
        (C6, C1, [[0.258819, 0, 0, -0.965926], [0.258819, 0, 0, 0.965926]]),
        (C3, C3, [[0.5, 0, 0, -0.866], [0.5, 0, 0, 0.866]]),
        (
            D2,
            C1,
            [
                [0.5**0.5, -(0.5**0.5), 0, 0],
                [0.5**0.5, 0, -(0.5**0.5), 0],
                [0.5**0.5, 0, 0, -(0.5**0.5)],
                [0.5**0.5, 0, 0, 0.5**0.5],
                [0.5**0.5, 0, 0.5**0.5, 0],
                [0.5**0.5, 0.5**0.5, 0, 0],
            ],
        ),
        (
            D3,
            C1,
            [
                [0.707107, -0.707107, 0, 0],
                [0.707107, -0.353553, -0.612372, 0],
                [0.707107, -0.353553, 0.612372, 0],
                [0.5, 0, 0, -0.866025],
                [0.5, 0, 0, 0.866025],
                [0.707107, 0.353553, -0.612372, 0],
                [0.707107, 0.353553, 0.612372, 0],
                [0.707107, 0.707107, 0, 0],
            ],
        ),
        (
            D6,
            C1,
            [
                [0.707107, -0.707107, 0, 0],
                [0.707107, -0.612372, -0.353553, 0],
                [0.707107, -0.612372, 0.353553, 0],
                [0.707107, -0.353553, -0.612372, 0],
                [0.707107, -0.353553, 0.612372, 0],
                [0.707107, 0, -0.707107, 0],
                [0.258819, 0, 0, -0.965926],
                [0.258819, 0, 0, 0.965926],
                [0.707107, 0, 0.707107, 0],
                [0.707107, 0.353553, -0.612372, 0],
                [0.707107, 0.353553, 0.612372, 0],
                [0.707107, 0.612372, -0.353553, 0],
                [0.707107, 0.612372, 0.353553, 0],
                [0.707107, 0.707107, 0, 0],
            ],
        ),
    ],
)
def test_get_large_cell_normals(s1, s2, expected):
    n = _get_large_cell_normals(s1, s2)
    assert np.allclose(n.data, expected, atol=1e-3)


def test_coverage_on_faces():
    o = OrientationRegion(Orientation([1, 1, 1, 1]))
    _ = o.faces()


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
    _ = get_proper_groups(Gl, Gr)


def test_get_proper_point_group_not_implemented():
    """Double inversion case not yet implemented."""
    with pytest.raises(NotImplementedError):
        _ = get_proper_groups(Csz, Csz)
