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

from orix.crystal_map import Phase
from orix.quaternion import OrientationRegion, Rotation, Symmetry, symmetry
from orix.quaternion.symmetry import get_point_group
from orix.sampling import sample_S2
from orix.sampling.SO3_sampling import _three_uniform_samples_method, uniform_SO3_sample
from orix.sampling._cubochoric_sampling import cubochoric_sampling
from orix.vector import Vector3d


def get_sample_fundamental(
    resolution: Union[int, float] = 2,
    point_group: Optional[Symmetry] = None,
    space_group: Optional[int] = None,
    phase: Optional[Phase] = None,
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
    phase
        The phase for which the fundamental zone rotations are sampled.
        If not given, the point group or space group is used to determine
        the crystal system.
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
    if phase is not None:
        point_group = phase.point_group
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
    phase: Phase = None,
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
    phase
        The phase for which the reduced fundamental zone rotations are
        sampled. If not given, the point group is used to determine the
        crystal system.
    point_group
        Symmetry operations that determines the unique directions. If ``Phase``
        is given the ``Phase.point_group`` is used instead. Defaults to
        no symmetry, which means sampling all 3D unit vectors.
    Returns
    -------
    Rotation
        (N, 3) array representing Euler angles for the different orientations
    """
    if point_group is None and phase is None:
        point_group = symmetry.C1
    if phase is not None:
        point_group = phase.point_group

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


def _corners_to_centroid_and_edge_centers(corners):
    """
    Produces the midpoints and center of a trio of corners
    Parameters
    ----------
    corners : list of lists
        Three corners of a streographic triangle
    Returns
    -------
    list_of_corners : list
        Length 7, elements ca, cb, cc, mean, cab, cbc, cac where naming is such that
        ca is the first corner of the input, and cab is the midpoint between
        corner a and corner b.
    """
    ca, cb, cc = corners[0], corners[1], corners[2]
    mean = tuple(np.add(np.add(ca, cb), cc))
    cab = tuple(np.add(ca, cb))
    cbc = tuple(np.add(cb, cc))
    cac = tuple(np.add(ca, cc))
    return [ca, cb, cc, mean, cab, cbc, cac]


def get_sample_zone_axis(
    density: str = "3",
    phase: Phase = None,
    return_directions: bool = False,
) -> Rotation:
    """Produces rotations to align various crystallographic directions with
    the sample zone axes.

    Parameters
    ----------
    density
        Either '3' or '7' for the number of directions to return.
    phase
        The phase for which the zone axis rotations are required.
    return_directions
        If True, returns the directions as well as the rotations.
    """
    system = phase.point_group.system
    corners_dict = {
        "cubic": [(0, 0, 1), (1, 0, 1), (1, 1, 1)],
        "hexagonal": [(0, 0, 1), (2, 1, 0), (1, 1, 0)],
        "orthorhombic": [(0, 0, 1), (1, 0, 0), (0, 1, 0)],
        "tetragonal": [(0, 0, 1), (1, 0, 0), (1, 1, 0)],
        "trigonal": [(0, 0, 1), (-1, -2, 0), (1, -1, 0)],
        "monoclinic": [(0, 0, 1), (0, 1, 0), (0, -1, 0)],
    }
    if density == "3":
        direction_list = corners_dict[system]
    elif density == "7":
        direction_list = _corners_to_centroid_and_edge_centers(corners_dict[system])
    else:
        raise ValueError("Density must be either 3 or 7")

    # rotate the directions to the z axis
    rots = np.stack(
        [
            Rotation.from_align_vectors(v, Vector3d.zvector()).data
            for v in direction_list
        ]
    )
    rotations = Rotation(rots)
    if return_directions:
        return rotations, direction_list
    else:
        return rotations
