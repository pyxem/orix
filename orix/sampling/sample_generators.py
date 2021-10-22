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

from orix.quaternion import OrientationRegion
from orix.quaternion.symmetry import get_point_group
from orix.sampling.SO3_sampling import uniform_SO3_sample, _three_uniform_samples_method
from orix.sampling._cubochoric_sampling import cubochoric_sampling


def get_sample_fundamental(
    resolution=2, point_group=None, space_group=None, method="cubochoric", **kwargs
):
    """Generates an equispaced grid of rotations within a fundamental
    zone.

    Parameters
    ----------
    resolution : float, optional
        The characteristic distance between a rotation and its neighbour
        in degrees.
    point_group : orix.quaternion.Symmetry, optional
        One of the 11 proper point groups. If not given, `space_group`
        must be.
    space_group: int, optional
        Between 1 and 231. Must be given if `point_group` is not.
    method : str, optional
        "cubochoric" (default), "haar_euler", or, "quaternion". See
        :func:`~orix.sampling.uniform_SO3_sample` for details.
    kwargs
        Keyword arguments passed on to the sampling method.

    Returns
    -------
    rot : ~orix.quaternion.Rotation
        Grid of rotations lying within the specified fundamental zone.

    See Also
    --------
    :func:`orix.sampling.uniform_SO3_sample`

    Examples
    --------
    >>> from orix.quaternion.symmetry import C2
    >>> from orix.sampling import get_sample_fundamental
    >>> rot = get_sample_fundamental(1, point_group=C2)
    """
    if point_group is None:
        point_group = get_point_group(space_group, proper=True)

    # TODO: provide some subspace selection options
    rot = uniform_SO3_sample(resolution, method=method, unique=False, **kwargs)

    fundamental_region = OrientationRegion.from_symmetry(point_group)
    rot = rot[rot < fundamental_region]

    rot = rot.unique()

    return rot


def get_sample_local(
    resolution=2, center=None, grid_width=10, method="cubochoric", **kwargs
):
    """Generates a grid of rotations about a given rotation.

    Parameters
    ----------
    resolution : float, optional
        The characteristic distance between a rotation and its neighbour
        in degrees.
    center : ~orix.quaternion.Rotation, optional
        The rotation at which the grid is centered. The identity is used
        if not given.
    grid_width : float, optional
        The largest angle of rotation in degrees away from center that
        is acceptable.
    method : str, optional
        "cubochoric", "haar_euler", or "quaternion". See
        :func:`~orix.sampling.uniform_SO3_sample` for details.
    kwargs
        Keyword arguments passed on to the sampling method.

    Returns
    -------
    r : ~orix.quaternion.Rotation
        Grid of rotations lying within `grid_width` of center.

    See Also
    --------
    :func:`orix.sampling.uniform_SO3_sample`
    """
    if method == "haar_euler":
        rot = uniform_SO3_sample(resolution, method=method, unique=False)
    elif method == "quaternion":
        rot = _three_uniform_samples_method(
            resolution, unique=False, max_angle=grid_width
        )
    else:  # method == "cubochoric"
        rot = cubochoric_sampling(resolution=resolution, **kwargs)

    rot = _remove_larger_than_angle(rot, grid_width)
    rot = rot.unique()

    if center is not None:
        rot = center * rot

    return rot


def _remove_larger_than_angle(r, max_angle):
    """Removes large angle rotations from a sample of rotations.

    Parameters
    ----------
    r : ~orix.quaternion.Rotation
        Sample of rotations.
    max_angle : float
        Maximum allowable angle (in degrees) from which a rotation can
        differ from the origin.

    Returns
    -------
    r : ~orix.quaternion.Rotation
        Rotations lying within the desired region.
    """
    half_angle = np.deg2rad(max_angle / 2)
    half_angles = np.arccos(r.a.data)  # Returns between 0 and pi
    mask = half_angles < half_angle
    r = r[mask]
    return r
