# -*- coding: utf-8 -*-
# Copyright 2018-2022 the orix developers
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

"""Generation of spherical grids in *S2*."""

from typing import Tuple

import numpy as np

from orix.vector import Vector3d


def _remove_pole_duplicates(
    azimuth: np.ndarray, polar: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove duplicate directions at the North (polar = 0) and South
    (polar = pi) poles from the grid on S2. In each case the direction
    with azimuth = 0 is kept.

    Parameters
    ----------
    azimuth, polar

    Returns
    -------
    azimuth, polar
    """
    mask_azimuth = azimuth > 0
    mask_polar_0 = np.isclose(polar, 0) * mask_azimuth
    mask_polar_pi = np.isclose(polar, np.pi) * mask_azimuth
    # create mask of vectors to keep
    mask = ~np.logical_or(mask_polar_0, mask_polar_pi)
    return azimuth[mask], polar[mask]


def _sample_S2_uv_mesh_arrays(
    resolution: float,
    hemisphere: str = "both",
    offset: float = 0,
    azimuthal_endpoint: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get spherical coordinates for UV mesh points on unit sphere *S2*.

    For more information see the docstring for
    :meth:`orix.sampling.S2_sampling.sample_S2_uv_mesh`.

    Parameters
    ----------
    resolution
        Maximum angle between nearest neighbour grid points, in degrees.
        The resolution of :math:`u` and :math:`v` are rounded up to get
        an integer number of equispaced polar and azimuthal grid lines.
    hemisphere
        Generate mesh points on either the "upper", "lower" or "both"
        hemispheres. Default is "both".
    offset
        Mesh points are offset in angular space by this fraction of the
        step size, must be in the range [0..1]. Default is 0.
    azimuthal_endpoint
        If True then endpoint of the azimuthal array is included in the
        calculation. Default is False.

    Returns
    -------
    azimuth, polar
    """
    hemisphere = hemisphere.lower()
    if hemisphere not in ("upper", "lower", "both"):
        raise ValueError('Hemisphere must be one of "upper", "lower", or "both".')

    if not 0 <= offset < 1:
        raise ValueError(
            "Offset is a fractional value of the angular step size "
            + "and must be in the range [0..1]."
        )

    if hemisphere == "both":
        polar_min = 0
        polar_max = 180
    elif hemisphere == "upper":
        polar_min = 0
        polar_max = 90
    elif hemisphere == "lower":
        polar_min = 90
        polar_max = 180
    polar_range = polar_max - polar_min
    # calculate steps in degrees to avoid rounding errors
    steps_azimuth = int(np.ceil(360 / resolution))
    steps_polar = int(np.ceil(polar_range / resolution)) + 1
    resolution = np.deg2rad(resolution)
    # calculate number of steps and step size angular spacing
    step_size_azimuth = (2 * np.pi) / steps_azimuth
    step_size_polar = np.deg2rad(polar_range) / (steps_polar - 1)

    azimuth = np.linspace(
        offset * step_size_azimuth,
        2 * np.pi + offset * step_size_azimuth,
        num=steps_azimuth,
        endpoint=azimuthal_endpoint,
    )
    # convert to radians
    polar_min, polar_max = np.deg2rad(polar_min), np.deg2rad(polar_max)
    polar = np.linspace(
        polar_min + offset * step_size_polar,
        polar_max + offset * step_size_polar,
        num=steps_polar,
        endpoint=True,
    )
    # polar coordinate cannot exceed polar_max
    polar = polar[polar <= polar_max]

    return azimuth, polar


def sample_S2_uv_mesh(
    resolution: float,
    hemisphere: str = "both",
    offset: float = 0,
    remove_pole_duplicates: bool = True,
) -> Vector3d:
    r"""Vectors of a UV mesh on a unit sphere *S2*.

    The mesh vertices are defined by the parametrization

    .. math::
        x &= \sin(u)\cos(v), \\
        y &= \sin(u)\sin(v), \\
        z &= \cos(u).

    Taken from diffsims.

    Parameters
    ----------
    resolution
        Maximum angle between nearest neighbour grid points, in degrees.
        The resolution of :math:`u` and :math:`v` are rounded up to get
        an integer number of equispaced polar and azimuthal grid lines.
    hemisphere
        Generate mesh points on the "upper", "lower" or "both"
        hemispheres. Default is "both".
    offset
        Mesh points are offset in angular space by this fraction of the
        step size, must be in the range [0..1]. Default is 0.
    remove_pole_duplicates
        If True the duplicate mesh grid points at the North and South
        pole of the unit sphere are removed. Default is True.

    Returns
    -------
    Vector3d
    """
    azimuth, polar = _sample_S2_uv_mesh_arrays(resolution, hemisphere, offset)
    azimuth_prod, polar_prod = np.meshgrid(azimuth, polar)

    if remove_pole_duplicates:
        azimuth_prod, polar_prod = _remove_pole_duplicates(azimuth_prod, polar_prod)

    return Vector3d.from_polar(azimuth=azimuth_prod, polar=polar_prod).unit


def _sample_S2_equal_area_arrays(
    resolution: float,
    hemisphere: str = "both",
    azimuthal_endpoint: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get spherical coordinates for equal area mesh points on unit
    sphere *S2*.

    For more information see the docstring for
    :meth:`orix.sampling.S2_sampling.sample_S2_equal_area_mesh`.

    Parameters
    ----------
    resolution
        The angular resolution in degrees of the azimuthal vectors.
    hemisphere
        Generate mesh points on the "upper", "lower" or "both"
        hemispheres. Default is "both".
    azimuthal_endpoint
        If True then endpoint of the azimuthal array is included in the
        calculation. Default is False.

    Returns
    -------
    azimuth, polar
    """
    hemisphere = hemisphere.lower()
    if hemisphere not in ("upper", "lower", "both"):
        raise ValueError('Hemisphere must be one of "upper", "lower", or "both".')

    # calculate number of steps and step size angular spacing
    # this parameter D in :cite:`rohrer2004distribution`.
    steps = int(np.ceil(90 / resolution))
    azimuth = np.linspace(0, 2 * np.pi, num=4 * steps, endpoint=azimuthal_endpoint)
    # polar coordinate is parameterized in terms of cos(theta)
    if hemisphere == "both":
        polar_min = 1
        polar_max = -1
        steps *= 2
    elif hemisphere == "upper":
        polar_min = 1
        polar_max = 0
    elif hemisphere == "lower":
        polar_min = 0
        polar_max = -1

    polar = np.linspace(
        polar_min,
        polar_max,
        num=steps,
        endpoint=True,
    )
    polar = np.arccos(polar)

    return azimuth, polar


def sample_S2_equal_area_mesh(
    resolution: float,
    hemisphere: str = "both",
    remove_pole_duplicates: bool = True,
) -> Vector3d:
    """Vectors of a cube mesh on a unit sphere *S2* according to equal
    area spacing :cite:`rohrer2004distribution`.

    Parameters
    ----------
    resolution
        The angular resolution in degrees of the azimuthal vectors.
    hemisphere
        Generate mesh points on the "upper", "lower" or "both"
        hemispheres. Default is "both".
    remove_pole_duplicates
        If True the duplicate mesh grid points at the North and South
        pole of the unit sphere are removed. If True then the returned
        vector has `ndim` = 1, whereas `ndim` = 2 (grid) if False.
        Default is True.

    Returns
    -------
    Vector3d
    """
    azimuth, polar = _sample_S2_equal_area_arrays(resolution, hemisphere)
    azimuth_prod, polar_prod = np.meshgrid(azimuth, polar)

    if remove_pole_duplicates:
        azimuth_prod, polar_prod = _remove_pole_duplicates(azimuth_prod, polar_prod)

    return Vector3d.from_polar(azimuth=azimuth_prod, polar=polar_prod).unit


def sample_S2_cube_mesh(
    resolution: float, grid_type: str = "spherified_corner"
) -> Vector3d:
    """Vectors of a cube mesh on a unit sphere *S2*.

    Taken from diffsims.

    Parameters
    ----------
    resolution
        Maximum angle between neighbour grid points, in degrees.
    grid_type
        Type of cube grid: "normalized", "spherified_edge" or
        "spherified_corner" (default).

    Returns
    -------
    Vector3d
    """
    vz = Vector3d.zvector()
    v011 = Vector3d((0, 1, 1))
    max_angle = vz.angle_with(v011)  # = pi / 4
    max_distance = vz.dot(v011)  # = 1

    res = np.radians(resolution)

    grid_type = grid_type.lower()
    grid_types = ["normalized", "spherified_edge", "spherified_corner"]
    if grid_type == grid_types[0]:
        grid_length = np.tan(res)
        steps = np.ceil(max_distance / grid_length)
        i = np.arange(-steps, steps) / steps
    elif grid_type == grid_types[1]:
        steps = np.ceil((max_angle / res).round(2))
        k = np.arange(-steps, steps)
        theta = np.arctan(max_distance) / steps
        i = np.tan(k * theta)
    elif grid_type == grid_types[2]:
        v111 = Vector3d((1, 1, 1))
        max_angle = vz.angle_with(v111)

        steps = np.ceil(max_angle / res)
        k = np.arange(-steps, steps)
        theta = np.arctan(np.sqrt(2)) / steps
        i = np.tan(k * theta) / np.sqrt(2)
    else:
        raise ValueError(
            f"The `grid_type` {grid_type} is not among the valid options {grid_types}"
        )

    x, y = np.meshgrid(i, i)
    x = x.ravel()
    y = y.ravel()
    z = np.ones(x.shape[0])

    # Grid on all faces of the cube, avoiding overlap of points on edges
    bottom = np.vstack([-x, -y, -z]).T
    top = np.vstack([x, y, z]).T
    east = np.vstack([z, x, -y]).T
    west = np.vstack([-z, -x, y]).T
    south = np.vstack([x, -z, y]).T
    north = np.vstack([-x, z, -y]).T

    # Two corners are missing with this procedure
    m_c = np.array([[-1, 1, 1], [1, -1, -1]])

    return Vector3d(np.vstack((bottom, top, east, west, south, north, m_c))).unit
