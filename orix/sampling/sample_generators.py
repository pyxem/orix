# -*- coding: utf-8 -*-
# Copyright 2018-2021 the orix developers
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

from orix.sampling.SO3_sampling import uniform_SO3_sample
from orix.quaternion.orientation_region import OrientationRegion
from orix.quaternion.symmetry import get_point_group


def get_sample_fundamental(resolution=2, point_group=None, space_group=None):
    """
    Generates an equispaced grid of rotations within a fundamental zone.

    Parameters
    ----------
    resolution : float, optional
        The characteristic distance between a rotation and its neighbour (degrees)
    point_group : orix.quaternion.symmetry.Symmetry, optional
        One of the 11 proper point groups, defaults to None
    space_group: int, optional
        Between 1 and 231, defaults to None

    Returns
    -------
    q : orix.quaternion.rotation.Rotation
        Grid of rotations lying within the specified fundamental zone

    See Also
    --------
    orix.sampling.utils.uniform_SO3_sample

    Examples
    --------
    >>> from orix.quaternion.symmetry import C2,C4
    >>> grid = get_sample_fundamental(1, point_group=C2)
    """
    if point_group is None:
        point_group = get_point_group(space_group, proper=True)

    # TODO: provide some subspace selection options
    q = uniform_SO3_sample(resolution)
    fundamental_region = OrientationRegion.from_symmetry(point_group)
    return q[q < fundamental_region]


def get_sample_local(resolution=2, center=None, grid_width=10,old_method=False):
    """
    Generates a grid of rotations about a given rotation

    Parameters
    ----------
    resolution : float, optional
        The characteristic distance between a rotation and its neighbour (degrees)
    center : orix.quaternion.rotation.Rotation, optional
        The rotation at which the grid is centered. If None (default) uses the identity
    grid_width : float, optional
        The largest angle of rotation away from center that is acceptable (degrees)
    old_method : bool, optional
        #TODO
    Returns
    -------
    q : orix.quaternion.rotation.Rotation
        Grid of rotations lying within grid_width of center

    See Also
    --------
    orix.sampling_utils.uniform_SO3_sample
    """
    if not old_method:
        q = uniform_SO3_sample(resolution, max_angle=None,old_method=False)
    else:
        q = uniform_SO3_sample(resolution,old_method=True)

    half_angle = np.deg2rad(grid_width / 2)
    half_angles = np.arccos(q.a.data)
    mask = np.logical_or(
        half_angles < half_angle, half_angles > (2 * np.pi - half_angle)
        )
    q = q[mask]

    if center is not None:
        q = center * q

    return q
