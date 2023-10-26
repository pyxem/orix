# -*- coding: utf-8 -*-
# Copyright 2018-2023 the orix developers
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
from orix.quaternion import symmetry
from orix.sampling import sample_S2


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


def get_sample_reduced_fundamental(
    resolution: float,
    mesh: str = None,
    point_group: Symmetry = None,
) -> Rotation:
    """Produces rotations to align various crystallographic directions with
    the z-axis, with the constraint that the first Euler angle phi_1=0.
    The crystallographic directions sample the fundamental zone, representing
    the smallest region of symmetrically unique directions of the relevant
    crystal system or point group.
    Parameters
    ----------
    resolution
        An angle in degrees representing the maximum angular distance to a
        first nearest neighbor grid point.
    mesh
        Type of meshing of the sphere that defines how the grid is created. See
        orix.sampling.sample_S2 for all the options. A suitable default is
        chosen depending on the crystal system.
        point_group
        Symmetry operations that determines the unique directions. Defaults to
        no symmetry, which means sampling all 3D unit vectors.
    Returns
    -------
    ConstrainedRotation
        (N, 3) array representing Euler angles for the different orientations
    """
    if point_group is None:
        point_group = symmetry.C1

    if mesh is None:
        s2_auto_sampling_map = {
            "triclinic": "icosahedral",
            "monoclinic": "icosahedral",
            "orthorhombic": "spherified_cube_edge",
            "tetragonal": "spherified_cube_edge",
            "cubic": "spherified_cube_edge",
            "trigonal": "hexagonal",
            "hexagonal": "hexagonal",
        }
        mesh = s2_auto_sampling_map[point_group.system]

    s2_sample = sample_S2(resolution, method=mesh)
    fundamental = s2_sample[s2_sample <= point_group.fundamental_sector]

    phi = fundamental.polar
    phi2 = (np.pi / 2 - fundamental.azimuth) % (2 * np.pi)
    phi1 = np.zeros(phi2.shape[0])
    euler_angles = np.vstack([phi1, phi, phi2]).T

    return Rotation.from_euler(euler_angles, degrees=False)


def get_sample_zone_axis(
    resolution: float,
    mesh: str = None,
    point_group: Symmetry = None,
) -> Rotation:
    """Produces the rotations to align various crystallographic directions with
    the z-axis, with the constraint that the first Euler angle phi_1=0.


    Parameters
    ----------
    resolution
        An angle in degrees representing the maximum angular distance to a
        first nearest neighbor grid point.
    mesh
        Type of meshing of the sphere that defines how the grid is created. See
        orix.sampling.sample_S2 for all the options. A suitable default is
        chosen depending on the crystal system.
        point_group
        Symmetry operations that determines the unique directions. Defaults to
        no symmetry, which means sampling all 3D unit vectors.
    Returns
    -------
    ConstrainedRotation
        (N, 3) array representing Euler angles for the different orientations
    """
    if point_group is None:
        point_group = symmetry.C1

    if mesh is None:
        s2_auto_sampling_map = {
            "triclinic": "icosahedral",
            "monoclinic": "icosahedral",
            "orthorhombic": "spherified_cube_edge",
            "tetragonal": "spherified_cube_edge",
            "cubic": "spherified_cube_edge",
            "trigonal": "hexagonal",
            "hexagonal": "hexagonal",
        }
        mesh = s2_auto_sampling_map[point_group.system]

    s2_sample = sample_S2(resolution, method=mesh)
    fundamental = s2_sample[s2_sample <= point_group.fundamental_sector]

    phi = fundamental.polar
    phi2 = (np.pi / 2 - fundamental.azimuth) % (2 * np.pi)
    phi1 = np.zeros(phi2.shape[0])
    euler_angles = np.vstack([phi1, phi, phi2]).T

    return Rotation.from_euler(euler_angles, degrees=False)
