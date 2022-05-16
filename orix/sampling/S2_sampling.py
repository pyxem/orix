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

from typing import Optional, Callable, Mapping, Any
from functools import partial

import numpy as np

from orix.vector import Vector3d
from orix.sampling._polyhedral_sampling import (
    _sample_length_equidistant,
    _edge_grid_normalized_cube,
    _edge_grid_spherified_edge_cube,
    _edge_grid_spherified_corner_cube,
    _compose_from_faces,
)


def sample_S2_uv_mesh(resolution: float) -> Vector3d:
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

    Returns
    -------
    vectors
        Vectors that sample the unit sphere.

    References
    ----------
    :cite:`cajaravelli2015four`
    """
    steps_azimuth = int(np.ceil(360 / resolution))
    steps_polar = int(np.ceil(180 / resolution)) + 1
    azimuth = np.linspace(0, 2 * np.pi, num=steps_azimuth, endpoint=False)
    polar = np.linspace(0, np.pi, num=steps_polar, endpoint=True)
    azimuth_prod, polar_prod = np.meshgrid(azimuth, polar)
    azimuth_prod = azimuth_prod.ravel()
    polar_prod = polar_prod.ravel()
    # remove duplicated vectors at north (polar == 0) and
    # south (polar == np.pi) poles. Keep the azimuth == 0 vector in each
    # case. Masks here are vectors to remove
    mask_azimuth = azimuth_prod > 0
    mask_polar_0 = np.isclose(polar_prod, 0) * mask_azimuth
    mask_polar_pi = np.isclose(polar_prod, np.pi) * mask_azimuth
    # create mask of vectors to keep
    mask = ~np.logical_or(mask_polar_0, mask_polar_pi)
    azimuth_prod = azimuth_prod[mask]
    polar_prod = polar_prod[mask]
    return Vector3d.from_polar(azimuth=azimuth_prod, polar=polar_prod).unit


def sample_S2_cube_mesh(
    resolution: float,
    grid_type: str = "spherified_edge",
) -> Vector3d:
    """Vectors of a cube mesh projected on a unit sphere *S2*.

    Parameters
    ----------
    resolution
        Maximum angle between neighbour grid points, in degrees.
    grid_type
        Type of cube grid: "normalized", "spherified_edge" (default) or
        "spherified_corner".

    Returns
    -------
    vectors
        Vectors that sample the unit sphere.

    Notes
    -----
    Vectors are sampled by projecting a grid on a cube onto the unit sphere.
    The mesh on the cube can be generated in a number of ways. A regular square
    grid with equidistant points corresponds to the 'normalized' option.
    'spherified_edge' corresponds to points such that the row of vectors from
    the [001] to [011] is equiangular. 'spherified_corner' corresponds to the case
    where the row of vectors from [001] to [111] is equiangular.

    References
    ----------
    :cite:`cajaravelli2015four`
    """
    grid_type = grid_type.lower()
    grid_mapping = {
        "normalized": _edge_grid_normalized_cube,
        "spherified_edge": _edge_grid_spherified_edge_cube,
        "spherified_corner": _edge_grid_spherified_corner_cube,
    }
    try:
        grid_on_edge = grid_mapping[grid_type](resolution)
    except KeyError:
        raise ValueError(
            f"The `grid_type` {grid_type} is not among the valid "
            f"options {list(grid_mapping.keys())}"
        )

    x, y = np.meshgrid(grid_on_edge, grid_on_edge)
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


def sample_S2_hexagonal_mesh(
    resolution: float,
) -> Vector3d:
    """Vectors of a hexagonal bipyramid mesh projected on a unit sphere *S2*.

    Parameters
    ----------
    resolution
        Maximum angle between neighbour grid points, in degrees.

    Returns
    -------
    vectors
        Vectors that sample the unit sphere.
    """
    number_of_steps = int(np.ceil(2 / np.tan(np.deg2rad(resolution))))
    if number_of_steps % 2 == 1:
        # an even number of steps is required to get a point in the middle
        # of the hexagon edge
        number_of_steps += 1
    grid_1D = _sample_length_equidistant(
        number_of_steps,
        length=1.0,
        include_start=True,
        include_end=True,
        positive_and_negative=False,
    )

    # top and bottom face of the hexagon
    axis_to_corner_1 = grid_1D[1:]
    axis_to_corner_2 = grid_1D
    u, v = np.meshgrid(axis_to_corner_1, axis_to_corner_2)
    u = u.ravel()
    v = v.ravel()

    # from square to hex lattice
    hexagon_edge_length = 2 / np.sqrt(3)
    transform = np.array([[hexagon_edge_length, hexagon_edge_length / 2], [0, 1]])
    uv = np.stack([u, v])
    xy = np.dot(transform, uv)
    x, y = xy

    # raise to pyramidal plane
    z = -1 / hexagon_edge_length * x - 1 / 2 * y + 1
    tolerance = -1e-7
    include_points = z > tolerance
    points_one_face = np.stack([coordinate[include_points] for coordinate in [x, y, z]])

    # repeat 6 times by rotating 60 degrees
    def rotation(r):
        return np.array(
            [[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]]
        )

    angle = np.deg2rad(60)
    top_faces = np.hstack(
        [np.dot(rotation(i * angle), points_one_face) for i in range(6)]
    )
    bottom_faces = top_faces.copy()
    bottom_faces[2] *= -1
    exclude_rim = bottom_faces[2] < tolerance
    bottom_faces = bottom_faces.T[exclude_rim].T
    north_pole = np.array([[0, 0, 1]]).T
    south_pole = np.array([[0, 0, -1]]).T
    all_points = np.hstack([top_faces, north_pole, bottom_faces, south_pole])

    return Vector3d(all_points.T).unit


def sample_S2_random_mesh(
    resolution: float,
    seed: Optional[int] = None,
) -> Vector3d:
    """Vectors of a random mesh on *S2*

    Parameters
    ----------
    resolution
        The expected mean angle between nearest neighbor
        grid points in degrees.
    seed
        Passed to :func:`numpy.random.default_rng`, defaults to None which
        will give a "new" random result each time.

    Returns
    -------
    vectors
        Vectors that sample the unit sphere.

    References
    ----------
    https://mathworld.wolfram.com/SpherePointPicking.html
    """
    # convert resolution in degrees to number of points
    number = int(1 / (4 * np.pi) * (360 / resolution) ** 2)
    rng = np.random.default_rng(seed=seed)
    xyz = rng.normal(size=(number, 3))
    return Vector3d(xyz).unit


def sample_S2_icosahedral_mesh(
    resolution: float,
) -> Vector3d:
    """Vectors of an icosahedral mesh on *S2*

    Parameters
    ----------
    resolution
        Maximum angle between neighbour grid points, in degrees.

    Returns
    -------
    vectors
        Vectors that sample the unit sphere.

    References
    ----------
    :cite:`meshzoo`
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0
    corners = np.array(
        [
            [-1, +t, +0],
            [+1, +t, +0],
            [-1, -t, +0],
            [+1, -t, +0],
            #
            [+0, -1, +t],
            [+0, +1, +t],
            [+0, -1, -t],
            [+0, +1, -t],
            #
            [+t, +0, -1],
            [+t, +0, +1],
            [-t, +0, -1],
            [-t, +0, +1],
        ]
    )
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    # icosahedron edge length
    a = np.linalg.norm(corners[0]) / np.sin(2 * np.pi / 5)
    # icosahedron inscribed sphere radius
    r_i = np.sqrt(3) / 12 * (3 + np.sqrt(5)) * a
    n = int(np.ceil(a / (r_i * np.tan(np.deg2rad(resolution)))))
    vertices = _compose_from_faces(corners, faces, n)
    return Vector3d(vertices).unit


SAMPLING_METHODS: Mapping[str, Callable] = {
    "uv": sample_S2_uv_mesh,
    "normalized_cube": partial(sample_S2_cube_mesh, grid_type="normalized"),
    "spherified_cube_edge": partial(sample_S2_cube_mesh, grid_type="spherified_edge"),
    "spherified_cube_corner": partial(
        sample_S2_cube_mesh, grid_type="spherified_corner"
    ),
    "icosahedral": sample_S2_icosahedral_mesh,
    "hexagonal": sample_S2_hexagonal_mesh,
    "random": sample_S2_random_mesh,
}
