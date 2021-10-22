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

"""Generation of spherical grids in *S2*."""

import numpy as np

from orix.vector import Vector3d


def sample_S2_uv_mesh(resolution):
    r"""Vectors of a UV mesh on a unit sphere *S2*.

    The mesh vertices are defined by the parametrization

    .. math::
        x &= \sin(u)\cos(v), \\
        y &= \sin(u)\sin(v), \\
        z &= \cos(u).

    Taken from diffsims.

    Parameters
    ----------
    resolution : float
        Maximum angle between nearest neighbour grid points, in degrees.
        The resolution of :math:`u` and :math:`v` are rounded up to get
        an integer number of equispaced polar and azimuthal grid lines.

    Returns
    -------
    Vector3d
    """
    steps_azimuth = int(np.ceil(360 / resolution))
    steps_polar = int(np.ceil(180 / resolution)) + 1
    azimuth = np.linspace(0, np.pi, num=steps_azimuth, endpoint=True)
    polar = np.linspace(0, 2 * np.pi, num=steps_polar, endpoint=False)
    azimuth_prod, polar_prod = np.meshgrid(azimuth, polar)
    azimuth_prod = azimuth_prod.ravel()
    polar_prod = polar_prod.ravel()
    return Vector3d.from_polar(azimuth=azimuth_prod, polar=polar_prod).unit


def sample_S2_cube_mesh(resolution, grid_type="spherified_corner"):
    """Vectors of a cube mesh on a unit sphere *S2*.

    Taken from diffsims.

    Parameters
    ----------
    resolution : float
        Maximum angle between neighbour grid points, in degrees.
    grid_type : str
        Type of cube grid: "normalized", "spherified_edge" or
        "spherified_corner" (default).

    Returns
    -------
    Vector3d
    """
    vz = Vector3d.zvector()
    v011 = Vector3d((0, 1, 1))
    max_angle = vz.angle_with(v011).data  # = pi / 4
    max_distance = vz.dot(v011).data  # = 1

    res = np.radians(resolution)

    grid_types = ["normalized", "spherified_edge", "spherified_corner"]
    if grid_type == grid_types[0]:
        grid_length = np.tan(res)
        steps = np.ceil(max_distance / grid_length)
        i = np.arange(-steps, steps) / steps
    elif grid_type == grid_types[1]:
        steps = np.ceil(max_angle / res)
        k = np.arange(-steps, steps)
        theta = np.arctan(max_distance) / steps
        i = np.tan(k * theta)
    elif grid_type == grid_types[2]:
        v111 = Vector3d((1, 1, 1))
        max_angle = vz.angle_with(v111).data

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
