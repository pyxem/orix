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

from orix.sampling.SO3_sampling import uniform_SO3_sample, _three_uniform_samples_method
from orix.quaternion.orientation_region import OrientationRegion
from orix.quaternion.symmetry import get_point_group


def get_sample_fundamental(
    resolution=2, point_group=None, space_group=None, method="harr_euler"
):
    """Generates an equispaced grid of rotations within a fundamental
    zone.

    Parameters
    ----------
    resolution : float, optional
        The characteristic distance between a rotation and its neighbour
        in degrees.
    point_group : orix.quaternion.Symmetry, optional
        One of the 11 proper point groups, defaults to None.
    space_group: int, optional
        Between 1 and 231, defaults to None.
    method : str, optional
        Either "harr_euler" (default) or "quaternion". See
        :func:`~orix.sampling.uniform_SO3_sample` for details.

    Returns
    -------
    r : orix.quaternion.Rotation
        Grid of rotations lying within the specified fundamental zone.

    See Also
    --------
    :func:`orix.sampling.uniform_SO3_sample`

    Examples
    --------
    >>> from orix.quaternion.symmetry import C2
    >>> from orix.sampling import get_sample_fundamental
    >>> grid = get_sample_fundamental(1, point_group=C2)
    """
    if point_group is None:
        point_group = get_point_group(space_group, proper=True)

    # TODO: provide some subspace selection options
    r = uniform_SO3_sample(resolution, method=method, unique=False)

    fundamental_region = OrientationRegion.from_symmetry(point_group)
    r = r[r < fundamental_region]

    r = r.unique()

    return r


def get_sample_local(resolution=2, center=None, grid_width=10, method="harr_euler"):
    """Generates a grid of rotations about a given rotation.

    Parameters
    ----------
    resolution : float, optional
        The characteristic distance between a rotation and its neighbour
        in degrees.
    center : orix.quaternion.Rotation, optional
        The rotation at which the grid is centered. If None (default)
        uses the identity.
    grid_width : float, optional
        The largest angle of rotation in degrees away from center that
        is acceptable.
    method : str, optional
        Either "harr_euler" (default) or "quaternion". See
        :func:`~orix.sampling.uniform_SO3_sample` for details.

    Returns
    -------
    r : orix.quaternion.Rotation
        Grid of rotations lying within `grid_width` of center.

    See Also
    --------
    :func:`orix.sampling.uniform_SO3_sample`
    """
    if method is not "quaternion":
        r = uniform_SO3_sample(resolution, method=method, unique=False)
    else:
        r = _three_uniform_samples_method(
            resolution, unique=False, max_angle=grid_width
        )

    r = _remove_larger_than_angle(r, grid_width)
    r = r.unique()

    if center is not None:
        r = center * r

    return r


def _remove_larger_than_angle(r, max_angle):
    """Removes large angle rotations from a sample of rotations.

    Parameters
    ----------
    r : orix.quaternion.Rotation
        Sample of rotations.
    max_angle : float
        Maximum allowable angle (in degrees) from which a rotation can
        differ from the origin.

    Returns
    -------
    r : orix.quaternion.Rotation
        Rotations lying within the desired region.
    """
    half_angle = np.deg2rad(max_angle / 2)
    half_angles = np.arccos(r.a.data)  # Returns between 0 and pi
    mask = half_angles < half_angle
    r = r[mask]
    return r
