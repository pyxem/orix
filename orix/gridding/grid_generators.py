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

""" This user facing code generates 'grids' in orientation spaces """

from orix.gridding.gridding_utils import (
    create_equispaced_grid,
    _get_proper_point_group,
)


def get_grid_fundamental(resolution, point_group=None, space_group=None):
    """
    Generates a grid of rotations that lie within a fundamental zone

    Parameters
    ----------
    resolution : float
        The smallest distance between a rotation and its neighbour (degrees)
    point_group : orix.symmettry
        One of the 11 proper point groups as
    space_group: int
        Between 1 and 231
    Returns
    -------

    See Also
    --------
    orix.gridding.utils.create_equispaced_grid

    Examples
    --------
    >>> from orix.quaternion.symmetry import C2,C4
    >>> grid = get_grid_fundamental(1,point_group=C2)
    >>> grid.shape()
    ...
    >>> grid = get_grid_fundamental(1,point_group=C4)
    >>> grid.shape()
    ...

    """
    if point_group is None:
        point_group = _get_proper_point_group(space_group)

    q = create_equispaced_grid(resolution)
    q = q < point_group

    return q


def get_grid_local(resolution, center, grid_width):
    """
    Generates a grid of rotations about a given rotation

    Parameters
    ----------
    resolution : float
        The smallest distance between a rotation and its neighbour (degrees)

    center :

    grid_width :
        The largest angle of rotation away from center that is acceptable (degrees)
    Returns
    -------

    See Also
    --------
    orix.gridding_utils.create_equispaced_grid
    """

    q = create_equispaced_grid(resolution)
    q = q[q.angle < grid_width]
    q = center * q #check this for rotation order
    return q
