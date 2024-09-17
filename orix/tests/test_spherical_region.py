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

import numpy as np
import pytest

from orix.vector import SphericalRegion, Vector3d


@pytest.fixture(params=[(0, 0, 1)])
def spherical_region(request):
    return SphericalRegion(request.param)


@pytest.fixture(params=[(0, 0, 1)])
def vector(request):
    return Vector3d(request.param)


@pytest.mark.parametrize(
    "spherical_region, vector, expected",
    [
        ([0, 0, 1], [[0, 0, 0.5], [0, 0, -0.5], [0, 1, 0]], [True, False, False]),
        ([[0, 0, 1], [0, 1, 0]], [[0, 1, 1], [0, 0, 1]], [True, False]),
    ],
    indirect=["spherical_region", "vector"],
)
def test_gt(spherical_region, vector, expected):
    inside = vector < spherical_region
    assert np.all(np.equal(inside, expected))


@pytest.mark.parametrize(
    "spherical_region, vector, expected",
    [
        ([0, 0, 1], [[0, 0, 0.5], [0, 0, -0.5], [0, 1, 0]], [True, False, True]),
        ([[0, 0, 1], [0, 1, 0]], [[0, 1, 1], [0, 0, 1]], [True, True]),
    ],
    indirect=["spherical_region", "vector"],
)
def test_ge(spherical_region, vector, expected):
    inside = vector <= spherical_region
    assert np.all(np.equal(inside, expected))
