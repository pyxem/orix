# -*- coding: utf-8 -*-
# Copyright 2018-2020 The pyXem developers
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

import numpy as np

from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C1, C2, C3, C4, C6, D2, D3, D4, D6, O, T
from orix.gridding.gridding_utils import (
    create_equispaced_grid,
    _get_proper_point_group,
)
from orix.gridding.grid_generators import get_grid_fundamental, get_grid_local


def test_get_proper_point_group():
    """ Makes sure all the ints from 1 to 230 give answers"""
    for _space_group in np.arange(1, 231):
        point_group = _get_proper_point_group(_space_group)
        assert point_group in [C1, C2, C3, C4, C6, D2, D3, D4, D6, O, T]


@pytest.fixture(scope="session")
def grid():
    return create_equispaced_grid(2)


@pytest.fixture(scope="session")
def fr():
    """ fixed rotation """
    r = Rotation([0.5, 0.5, 0, 0])
    return r


def test_create_equispaced_grid_regions(grid, fr):
    """ Checks that different regions have the same density"""
    around_zero = grid[grid.a > 0.9]
    moved = fr * grid
    elsewhere = moved[grid.a > 0.9]
    # extra line simplifies the stacktrack
    x, y = around_zero.size, elsewhere.size
    assert np.isclose(x, y, rtol=0.01)


def test_create_equispaced_grid_resolution(grid):
    """ Checks that doubling resolution doubles density (8-fold counts) """
    lower = create_equispaced_grid(4)
    x, y = lower.size * 8, grid.size
    assert np.isclose(x, y, rtol=0.01)


def test_get_grid_local_width(fr):
    """ Checks that doubling the width 8 folds the number of points """
    x = get_grid_local(np.pi, fr, 15).size * 8
    y = get_grid_local(np.pi, fr, 30).size
    assert np.isclose(x, y, rtol=0.01)


@pytest.fixture(scope="session")
def C6_grid():
    return get_grid_fundamental(4, point_group=C6)


def test_get_grid_fundamental_zone_order(C6_grid):
    """ Cross check point counts to group order terms """
    D6_grid = get_grid_fundamental(4, point_group=D6)
    ratio = C6_grid.size / D6_grid.size
    assert np.isclose(ratio, 2, rtol=0.01)


def test_get_grid_fundamental_space_group(C6_grid):
    """ Going via the space_group route """
    # assert that space group #3 is has pg C2
    assert C2 == _get_proper_point_group(3)
    C2_grid = get_grid_fundamental(4, space_group=3)
    ratio = C2_grid.size / C6_grid.size
    assert np.isclose(ratio, 3, rtol=0.01)
