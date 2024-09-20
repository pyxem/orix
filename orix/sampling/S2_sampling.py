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

"""Generation of spherical grids in *S2*."""

from functools import partial
from typing import Callable, List, Mapping, Optional, Tuple

import numpy as np

from orix.sampling._polyhedral_sampling import (
    _compose_from_faces,
    _edge_grid_normalized_cube,
    _edge_grid_spherified_corner_cube,
    _edge_grid_spherified_edge_cube,
    _sample_length_equidistant,
)
from orix.vector import Vector3d


def _remove_pole_duplicates(
    azimuth: np.ndarray, polar: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove duplicate directions at the North (polar = 0) and South
    (polar = pi) poles from the grid on S2. In each case the direction
    with azimuth = 0 is kept.

    Parameters
    ----------
    azimuth
        Azimuth angles.
    polar
        Polar angles.

    Returns
    -------
    azimuth
        Azimuth angles without duplicates.
    polar
        Polar angles without duplicates.
    """
    mask_azimuth = azimuth > 0
    mask_polar_0 = np.isclose(polar, 0) * mask_azimuth
    mask_polar_pi = np.isclose(polar, np.pi) * mask_azimuth
    # create mask of vectors to keep
    mask = ~np.logical_or(mask_polar_0, mask_polar_pi)
    return azimuth[mask], polar[mask]


def _sample_S2_uv_mesh_coordinates(
    resolution: float,
    hemisphere: str = "both",
    offset: float = 0,
    azimuth_endpoint: bool = False,
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
        Generate mesh points on either the ``"upper"``, ``"lower"`` or
        ``"both"`` hemispheres. Default is ``"both"``.
    offset
        Mesh points are offset in angular space by this fraction of the
        step size, must be in the range [0..1]. Default is 0.
    azimuth_endpoint
        If ``True`` then endpoint of the azimuth array is included in
        the calculation. Default is ``False``.

    Returns
    -------
    azimuth
        Azimuth angles.
    polar
        Polar angles.
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
        endpoint=azimuth_endpoint,
    )
    # convert to radians
    polar_min, polar_max = np.deg2rad(polar_min), np.deg2rad(polar_max)
    polar = np.linspace(
        polar_min + offset * step_size_polar,
        polar_max + offset * step_size_polar,
        num=steps_polar,
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
    r"""Return vectors of a UV mesh on a unit sphere *S2*
    :cite:`cajaravelli2015four`.

    The mesh vertices are defined by the parametrization

    .. math::
        x &= \sin(u)\cos(v), \\
        y &= \sin(u)\sin(v), \\
        z &= \cos(u).

    Parameters
    ----------
    resolution
        Maximum angle between nearest neighbour grid points, in degrees.
        The resolution of :math:`u` and :math:`v` are rounded up to get
        an integer number of equispaced polar and azimuthal grid lines.
    hemisphere
        Generate mesh points on the ``"upper"``, ``"lower"`` or
        ``"both"`` hemispheres. Default is ``"both"``.
    offset
        Mesh points are offset in angular space by this fraction of the
        step size, must be in the range [0..1]. Default is 0.
    remove_pole_duplicates
        If ``True`` the duplicate mesh grid points at the North and
        South pole of the unit sphere are removed. Default is ``True``.

    Returns
    -------
    vec
        Vectors that sample the unit sphere.
    """
    azimuth, polar = _sample_S2_uv_mesh_coordinates(resolution, hemisphere, offset)
    azimuth_prod, polar_prod = np.meshgrid(azimuth, polar)

    if remove_pole_duplicates:
        azimuth_prod, polar_prod = _remove_pole_duplicates(azimuth_prod, polar_prod)

    return Vector3d.from_polar(azimuth=azimuth_prod, polar=polar_prod).unit


def _sample_S2_equal_area_coordinates(
    resolution: float,
    hemisphere: str = "both",
    azimuth_endpoint: bool = False,
    azimuth_range: Optional[Tuple[float, float]] = None,
    polar_range: Optional[Tuple[float, float]] = None,
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
        Generate mesh points on the ``"upper"``, ``"lower"`` or
        ``"both"`` hemispheres. Default is ``"both"``.
    azimuth_endpoint
        If ``True`` then endpoint of the azimuth array is included in
        the calculation. Default is ``False``.
    azimuth_range, polar_range
        The (min, max) angular range for the azimuthal and polar
        coordinates, respectively, in radians. If provided then the
        ``hemisphere`` argument is ignored. Default is ``None``.

    Returns
    -------
    azimuth
        Azimuth angles.
    polar
        Polar angles.
    """
    # calculate number of steps and step size angular spacing
    # this parameter D in :cite:`rohrer2004distribution`.
    steps = int(np.ceil(90 / resolution))

    if azimuth_range is not None:
        azimuth_min, azimuth_max = azimuth_range
        if azimuth_min >= azimuth_max:
            raise ValueError(
                "`azimuth_range` requires values (min, max) where min < max."
            )
    else:
        # use full range
        azimuth_min, azimuth_max = 0, 2 * np.pi

    # no wrap around
    azimuth_min = max(azimuth_min, 0)
    azimuth_max = min(azimuth_max, 2 * np.pi)

    azimuth_range = azimuth_max - azimuth_min
    # azimuth should have 4D steps over range [0..2pi]
    azimuth_num = int(np.ceil(azimuth_range / (np.pi / 2) * steps))

    # polar coordinate is parameterized in terms of cos(theta)
    if polar_range is not None:
        polar_min, polar_max = polar_range
        # no wrap around
        polar_min = max(polar_min, 0)
        polar_max = min(polar_max, np.pi)
        if polar_min >= polar_max:
            raise ValueError(
                "`polar_range` requires values (min, max) where min < max."
            )
        # convert to units of cos(theta) for equal area spacing
        polar_min, polar_max = np.cos((polar_min, polar_max))
    else:
        hemisphere = hemisphere.lower()
        if hemisphere not in ("upper", "lower", "both"):
            raise ValueError("`hemisphere` must be one of 'upper', 'lower', or 'both'.")
        # polar_min and polar_max in units of cos(theta)
        if hemisphere == "both":
            polar_min = 1
            polar_max = -1
        elif hemisphere == "upper":
            polar_min = 1
            polar_max = 0
        elif hemisphere == "lower":
            polar_min = 0
            polar_max = -1

    polar_range = polar_min - polar_max  # opposite as cos([0..pi]) -> [1..-1]
    # polar should have D steps over range [0..pi/2] rad, ie. [1..0]
    # extra point as polar endpoint is True
    polar_num = int(np.ceil(polar_range * steps)) + 1

    # extra data point to account for endpoint
    if azimuth_endpoint:
        azimuth_num += 1

    azimuth = np.linspace(
        azimuth_min, azimuth_max, num=azimuth_num, endpoint=azimuth_endpoint
    )
    polar = np.linspace(
        polar_min,
        polar_max,
        num=polar_num,
    )
    polar = np.arccos(polar)

    return azimuth, polar


def sample_S2_equal_area_mesh(
    resolution: float,
    hemisphere: str = "both",
    remove_pole_duplicates: bool = True,
) -> Vector3d:
    """Return vectors of a cube mesh on a unit sphere *S2* according to
    equal area spacing :cite:`rohrer2004distribution`.

    Parameters
    ----------
    resolution
        The angular resolution in degrees of the azimuthal vectors.
    hemisphere
        Generate mesh points on the ``"upper"``, ``"lower"`` or
        ``"both"`` hemispheres. Default is ``"both"``.
    remove_pole_duplicates
        If ``True`` the duplicate mesh grid points at the North and
        South pole of the unit sphere are removed. If ``True`` then the
        returned vector has ``ndim = 1``, whereas ``ndim = 2`` (grid) if
        ``False``. Default is ``True``.

    Returns
    -------
    vec
        Vectors that sample the unit sphere.
    """
    azimuth, polar = _sample_S2_equal_area_coordinates(resolution, hemisphere)
    azimuth_prod, polar_prod = np.meshgrid(azimuth, polar)

    if remove_pole_duplicates:
        azimuth_prod, polar_prod = _remove_pole_duplicates(azimuth_prod, polar_prod)

    return Vector3d.from_polar(azimuth=azimuth_prod, polar=polar_prod).unit


def sample_S2_cube_mesh(
    resolution: float, grid_type: str = "spherified_corner"
) -> Vector3d:
    """Return vectors of a cube mesh projected on a unit sphere *S2*
    :cite:`cajaravelli2015four`.

    Parameters
    ----------
    resolution
        Maximum angle between neighbour grid points, in degrees.
    grid_type
        Type of cube grid: ``"normalized"``, ``"spherified_edge"`` or
        ``"spherified_corner"``.

    Returns
    -------
    vec
        Vectors that sample the unit sphere.

    Notes
    -----
    Vectors are sampled by projecting a grid on a cube onto the unit
    sphere. The mesh on the cube can be generated in a number of ways. A
    regular square grid with equidistant points corresponds to the
    ``"normalized"`` option. ``"spherified_edge"`` corresponds to points
    such that the row of vectors from the [001] to [011] is equiangular.
    ``"spherified_corner"`` corresponds to the case where the row of
    vectors from [001] to [111] is equiangular.
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
            f"The `grid_type` {grid_type} is not among the valid options "
            f"{list(grid_mapping.keys())}"
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


def sample_S2_hexagonal_mesh(resolution: float) -> Vector3d:
    """Return vectors of a hexagonal bipyramid mesh projected on a unit
    sphere *S2*.

    Parameters
    ----------
    resolution
        Maximum angle between neighbour grid points, in degrees.

    Returns
    -------
    vec
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
    def rotation(r: np.ndarray) -> np.ndarray:
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


def sample_S2_random_mesh(resolution: float, seed: Optional[int] = None) -> Vector3d:
    """Return vectors of a random mesh on *S2*.

    Parameters
    ----------
    resolution
        The expected mean angle between nearest neighbor grid points in
        degrees.
    seed
        Passed to :func:`numpy.random.default_rng`, defaults to None
        which will give a "new" random result each time.

    Returns
    -------
    vec
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


def sample_S2_icosahedral_mesh(resolution: float) -> Vector3d:
    """Return vectors of an icosahedral mesh on *S2* :cite:`meshzoo`.

    Parameters
    ----------
    resolution
        Maximum angle between neighbour grid points, in degrees.

    Returns
    -------
    vec
        Vectors that sample the unit sphere.
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


_sampling_method_registry: Mapping[str, Callable] = {
    "uv": sample_S2_uv_mesh,
    "equal_area": sample_S2_equal_area_mesh,
    "normalized_cube": partial(sample_S2_cube_mesh, grid_type="normalized"),
    "spherified_cube_edge": partial(sample_S2_cube_mesh, grid_type="spherified_edge"),
    "spherified_cube_corner": partial(
        sample_S2_cube_mesh, grid_type="spherified_corner"
    ),
    "icosahedral": sample_S2_icosahedral_mesh,
    "hexagonal": sample_S2_hexagonal_mesh,
    "random": sample_S2_random_mesh,
}
sampling_methods: List[str] = []
_sampling_method_names = set()
for sampling_name, sampling_method in _sampling_method_registry.items():
    sampling_methods.append(sampling_name)
    _func = (
        sampling_method.func
        if isinstance(sampling_method, partial)
        else sampling_method
    )
    _sampling_method_names.add(f":func:`orix.sampling.{_func.__name__}`")

_s2_sampling_docstring = (
    """Return unit vectors that sample S2 with a specific angular
    resolution.

    Parameters
    ----------
    resolution
        Maximum angle between nearest neighbour grid points, in degrees.
    method
        Sphere meshing method. Options are: {}. The default is
        ``\"spherified_cube_edge\"``.
    **kwargs
        Keyword arguments passed to the sampling function. For details
        see the sampling functions listed below.

    Returns
    -------
    vec
        Vectors that sample the unit sphere.

    See Also
    --------
    {}
    """
).format(
    ", ".join(map(lambda x: f'``"{x}"``', sampling_methods)),
    "\n    ".join(_sampling_method_names),
)


def sample_S2(
    resolution: float, method: str = "spherified_cube_edge", **kwargs
) -> Vector3d:
    try:
        sampling_method = _sampling_method_registry[method]
    except KeyError:
        raise NotImplementedError(
            f"Method not implemented. Valid options: {sampling_methods}"
        )
    return sampling_method(resolution, **kwargs)


setattr(sample_S2, "__doc__", _s2_sampling_docstring)
