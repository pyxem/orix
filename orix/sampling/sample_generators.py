# -*- coding: utf-8 -*-
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

from typing import Optional, Union

import numpy as np

from orix.quaternion import OrientationRegion, Rotation, Symmetry
from orix.quaternion.symmetry import get_point_group
from orix.sampling.SO3_sampling import _three_uniform_samples_method, uniform_SO3_sample
from orix.sampling._cubochoric_sampling import cubochoric_sampling


def get_sample_fundamental(
    resolution: Union[int, float] = 2,
    point_group: Optional[Symmetry] = None,
    space_group: Optional[int] = None,
    method: str = "cubochoric",
    **kwargs
) -> Rotation:
    """Return an equispaced grid of rotations within a fundamental zone.

    Parameters
    ----------
    resolution
        The characteristic distance between a rotation and its neighbour
        in degrees.
    point_group
        One of the 11 proper point groups. If not given, ``space_group``
        must be.
    space_group
        Between 1 and 231. Must be given if ``point_group`` is not.
    method
        ``"cubochoric"`` (default), ``"haar_euler"`` or
        ``"quaternion"``. See :func:`~orix.sampling.uniform_SO3_sample`
        for details.
    **kwargs
        Keyword arguments passed on to the sampling method.

    Returns
    -------
    rot
        Grid of rotations lying within the specified fundamental zone.

    See Also
    --------
    orix.sampling.uniform_SO3_sample

    Examples
    --------
    >>> from orix.quaternion.symmetry import Oh
    >>> from orix.sampling import get_sample_fundamental
    >>> rot = get_sample_fundamental(5, point_group=Oh)
    >>> rot
    Rotation (6579,)
    [[ 0.877  -0.2774 -0.2774 -0.2774]
     [ 0.877  -0.2884 -0.2884 -0.2538]
     [ 0.877  -0.2986 -0.2986 -0.2291]
     ...
     [ 0.877   0.2986  0.2986  0.2291]
     [ 0.877   0.2884  0.2884  0.2538]
     [ 0.877   0.2774  0.2774  0.2774]]
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
    resolution: Union[int, float] = 2,
    center: Optional[Rotation] = None,
    grid_width: Union[int, float] = 10,
    method: str = "cubochoric",
    **kwargs
) -> Rotation:
    """Return a grid of rotations about a given rotation.

    Parameters
    ----------
    resolution
        The characteristic distance between a rotation and its neighbour
        in degrees.
    center
        The rotation at which the grid is centered. The identity is used
        if not given.
    grid_width
        The largest angle of rotation in degrees away from center that
        is acceptable.
    method
        ``"cubochoric"`` (default), ``"haar_euler"`` or
        ``"quaternion"``. See :func:`~orix.sampling.uniform_SO3_sample`
        for details.
    **kwargs
        Keyword arguments passed on to the sampling method.

    Returns
    -------
    rot
        Grid of rotations lying within ``grid_width`` of center.

    See Also
    --------
    orix.sampling.uniform_SO3_sample
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


def _remove_larger_than_angle(rot: Rotation, max_angle: Union[int, float]) -> Rotation:
    """Remove large angle rotations from a sample of rotations.

    Parameters
    ----------
    rot
        Sample of rotations.
    max_angle
        Maximum allowable angle (in degrees) from which a rotation can
        differ from the origin.

    Returns
    -------
    rot_out
        Rotations lying within the desired region.
    """
    half_angle = np.deg2rad(max_angle / 2)
    half_angles = np.arccos(rot.a)  # Returns between 0 and pi
    mask = half_angles < half_angle
    rot_out = rot[mask]
    return rot_out
