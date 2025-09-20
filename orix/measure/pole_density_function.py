#
# Copyright 2018-2025 the orix developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix. If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from orix.projections.stereographic import StereographicProjection
from orix.quaternion.symmetry import C1, Symmetry
from orix.sampling.S2_sampling import _sample_S2_uv_mesh_coordinates
from orix.vector.vector3d import Vector3d


def pole_density_function(
    *args: np.ndarray | Vector3d,
    resolution: float = 1,
    sigma: float = 5,
    weights: np.ndarray | None = None,
    hemisphere: str = "upper",
    symmetry: Symmetry | None = C1,
    log: bool = False,
    mrd: bool = True,
) -> tuple[np.ma.MaskedArray, tuple[np.ndarray, np.ndarray]]:
    """Compute the Pole Density Function (PDF) of vectors on a
    cubed-sphere grid and return a map in polar coordinates for
    plotting.

    If ``symmetry`` is defined then the PDF is symmetrizes and the
    density map is returned only on the fundamental sector.

    Parameters
    ----------
    args
        Vector(s), or azimuth and polar angles of the vectors, the
        latter passed as two separate arguments.
    resolution
        The angular resolution of the sampling grid in degrees.
        Default value is 1.
    sigma
        The angular resolution of the applied broadening in degrees.
        Default value is 5.
    weights
        The weights for the individual vectors. Default is ``None``, in
        which case the weight of each vector is 1.
    hemisphere
        Which hemisphere(s) to plot the vectors on, options are
        ``"upper"`` and ``"lower"``. Default is ``"upper"``.
    symmetry
        If provided the PDF is calculated within the fundamental sector
        of the point group symmetry, otherwise the PDF is calculated
        on ``hemisphere``. Default is ``None``.
    log
        If ``True`` the log(PDF) is calculated. Default is ``True``.
    mrd
        If ``True`` the returned PDF is in units of Multiples of Random
        Distribution (MRD), otherwise the units are bin counts. Default
        is ``True``.

    Returns
    -------
    hist
        The computed histogram, shape is (N, M).
    x, y
        Tuple of coordinate grids for the bin edges of ``hist``. The
        units of ``x`` and ``y`` are cartesian coordinates on the
        stereographic projection plane and the shape of both ``x`` and
        ``y`` is (N + 1, M + 1).

    See Also
    --------
    orix.plot.InversePoleFigurePlot.pole_density_function
    orix.plot.StereographicPlot.pole_density_function
    orix.vector.Vector3d.pole_density_function
    """

    hemisphere = hemisphere.lower()
    poles = {"upper": -1, "lower": 1}
    sp = StereographicProjection(poles[hemisphere])

    # If user explicitly passes symmetry=None
    if symmetry is None:
        symmetry = C1

    if len(args) == 1:
        v = args[0]
        if not isinstance(v, Vector3d):
            raise TypeError(
                "If one argument is passed it must be an instance of "
                + "`orix.vector.Vector3d`."
            )
    elif len(args) == 2:
        # azimuth and polar angles
        v = Vector3d.from_polar(*args)
    else:
        raise ValueError(
            "Accepts only one (Vector3d) or two (azimuth, polar)\
                input arguments."
        )

    if v.size == 0:
        raise ValueError

    if np.any(v.norm) == 0.0:
        raise ValueError

    # IF we blur by a lot, save some compute time by doing the histograms
    # at lower resolution.
    if resolution < 0.2 * sigma and resolution < 1.0:
        histogram_resolution = 0.2 * sigma
        plot_resolution = resolution
    else:
        histogram_resolution = resolution
        plot_resolution = resolution

    # Do actual histogramming
    bins = int(90 / histogram_resolution)
    bin_edges = np.linspace(-np.pi / 4, np.pi / 4, bins + 1)
    hist = np.zeros((6, bins, bins))

    # Explicit symmetrization
    for rotation in symmetry:

        face_index_array, face_coordinates = _cube_gnom_coordinates(rotation * v)
        face_index_array = face_index_array.ravel()
        face_coordinate_1 = face_coordinates[0].ravel()
        face_coordinate_2 = face_coordinates[1].ravel()
        for face_index in range(6):
            this_face = face_index_array == face_index
            w = weights[this_face] if weights is not None else None
            hist[face_index] += (
                np.histogramdd(
                    (face_coordinate_1[this_face], face_coordinate_2[this_face]),
                    bins=(bin_edges, bin_edges),
                    weights=w,
                )[0]
                / symmetry.size
            )

    # Bins are not all same solid angle area, so we need to normalize.
    if mrd:
        bin_middles = np.linspace(
            -np.pi / 4 + np.pi / 4 / bins, np.pi / 4 - np.pi / 4 / bins, bins
        )
        y_ang, z_ang = np.meshgrid(bin_middles, bin_middles)
        solid_angle_term = (
            1
            / (np.tan(y_ang) ** 2 + np.tan(z_ang) ** 2 + 1)
            / (np.cos(y_ang) * np.cos(z_ang))
            / (1 - 0.5 * (np.sin(z_ang) * np.sin(y_ang)) ** 2)
        )
        solid_angle_term *= 1 / 6 / np.sum(solid_angle_term)
        if weights is not None:
            solid_angle_term *= np.sum(weights)
        else:
            solid_angle_term *= v.size
        hist = hist / solid_angle_term[np.newaxis, ...]

    # Smoothing
    if sigma != 0.0:
        # If smoothing is only a bit, du 60 small steps,
        # otherwise do a max step-size.
        if (sigma / histogram_resolution) ** 2 <= 20:
            N = 60
            t = 1 / N * (sigma / histogram_resolution) ** 2
        else:
            t = 1 / 3
            N = int(1 / t * (sigma / histogram_resolution) ** 2)

        hist = _smooth_gnom_cube_histograms(hist, t, N)

    # Make plot grid
    azimuth_coords, polar_coords = _sample_S2_uv_mesh_coordinates(
        plot_resolution,
        hemisphere="upper",
        azimuth_endpoint=True,
    )
    azimuth_grid, polar_grid = np.meshgrid(
        azimuth_coords,
        polar_coords,
        indexing="ij",
    )
    azimuth_center_grid, polar_center_grid = np.meshgrid(
        azimuth_coords[:-1] + np.diff(azimuth_coords) / 2,
        polar_coords[:-1] + np.diff(polar_coords) / 2,
        indexing="ij",
    )

    v_grid = Vector3d.from_polar(
        azimuth=azimuth_center_grid, polar=polar_center_grid
    ).unit
    v_grid_vertexes = Vector3d.from_polar(azimuth=azimuth_grid, polar=polar_grid).unit

    mask = ~(v_grid <= symmetry.fundamental_sector)
    v_grid = v_grid.in_fundamental_sector(symmetry)
    v_grid_vertexes = v_grid_vertexes.in_fundamental_sector(symmetry)

    # Interpolation from histograms to plot grid
    grid_face_index, grid_face_coords = _cube_gnom_coordinates(v_grid)
    hist_grid = np.zeros(v_grid.shape)
    bin_middles = np.linspace(
        -np.pi / 4 + np.pi / 4 / bins, np.pi / 4 - np.pi / 4 / bins, bins
    )
    for face_index in range(6):
        interpolator = RegularGridInterpolator(
            (bin_middles, bin_middles),
            hist[face_index],
            bounds_error=False,
            fill_value=None,
        )
        this_face = grid_face_index == face_index
        hist_grid[this_face] = interpolator(grid_face_coords[:, this_face].T)
    hist = hist_grid

    # Mask out points outside funamental region.
    hist = np.ma.array(hist, mask=mask)

    # Transform grdi to mystery coordinates used by plotting routine
    x, y = sp.vector2xy(v_grid_vertexes)
    x, y = x.reshape(v_grid_vertexes.shape), y.reshape(v_grid_vertexes.shape)

    if log:
        # +1 to avoid taking the log of 0
        hist = np.log(hist + 1)

    return hist, (x, y)


def _cube_gnom_coordinates(
    vectors: Vector3d,
) -> tuple[np.ndarray[int], np.ndarray[float]]:
    """Assigns an index (0 to 5) to an array of ```Vector3d```
    assigning them each to a face of a cube in the followiung way:

    Index 0 is positive x face. (Includes +x+y and +x+z edges.)
    Index 1 is negative x face. (Includes -x-y and -x-z edges.)
    Index 2 is positive y face. (Includes +y+z and +y-x edges.)
    Index 3 is negative y face. (Includes -y-z and -y+x edges.)
    Index 4 is positive z face. (Includes -y+z and -x+z edges.)
    Index 5 is negative z face. (Includes +y-z and +x-z edges.)

    The two "extra" corners are assigned to Index 0 and Index 1 respectively.

    Then each point is given 2D coordinates on the respective
    faces. Coordinates are in angles wrt. cube face center-lines.
    Always ordered with increasing coordinates.
    First coordinate comes first x -> y -> z.
    """

    if np.any(vectors.norm == 0.0):
        raise ValueError

    # Assign face index to each vector
    face_index = np.zeros(vectors.shape, dtype=int)

    indx = np.all(
        [
            vectors.x >= vectors.y,
            vectors.x >= -vectors.y,
            vectors.x >= vectors.z,
            vectors.x >= -vectors.z,
        ],
        axis=0,
    )
    face_index[indx] = 0

    indx = np.all(
        [
            vectors.x <= vectors.y,
            vectors.x <= -vectors.y,
            vectors.x <= vectors.z,
            vectors.x <= -vectors.z,
        ],
        axis=0,
    )
    face_index[indx] = 1

    indx = np.all(
        [
            vectors.y > vectors.x,
            vectors.y >= -vectors.x,
            vectors.y >= vectors.z,
            vectors.y > -vectors.z,
        ],
        axis=0,
    )
    face_index[indx] = 2

    indx = np.all(
        [
            vectors.y < vectors.x,
            vectors.y <= -vectors.x,
            vectors.y <= vectors.z,
            vectors.y < -vectors.z,
        ],
        axis=0,
    )
    face_index[indx] = 3

    indx = np.all(
        [
            vectors.z > vectors.x,
            vectors.z >= -vectors.x,
            vectors.z > vectors.y,
            vectors.z >= -vectors.y,
        ],
        axis=0,
    )
    face_index[indx] = 4

    indx = np.all(
        [
            vectors.z < vectors.x,
            vectors.z <= -vectors.x,
            vectors.z < vectors.y,
            vectors.z <= -vectors.y,
        ],
        axis=0,
    )
    face_index[indx] = 5

    # Assign coordinates
    coordinates = np.zeros((2,) + vectors.shape)
    unit_vectors = vectors.unit

    #  Comment: no need for np.arctan2. We know denom is pos
    #  so np.arctan should be faster.
    this_face = face_index == 0
    coordinates[0, this_face] = np.arctan(
        unit_vectors.y[this_face] / unit_vectors.x[this_face]
    )
    coordinates[1, this_face] = np.arctan(
        unit_vectors.z[this_face] / unit_vectors.x[this_face]
    )

    this_face = face_index == 1
    coordinates[0, this_face] = np.arctan(
        -unit_vectors.y[this_face] / unit_vectors.x[this_face]
    )
    coordinates[1, this_face] = np.arctan(
        -unit_vectors.z[this_face] / unit_vectors.x[this_face]
    )

    this_face = face_index == 2
    coordinates[0, this_face] = np.arctan(
        unit_vectors.x[this_face] / unit_vectors.y[this_face]
    )
    coordinates[1, this_face] = np.arctan(
        unit_vectors.z[this_face] / unit_vectors.y[this_face]
    )

    this_face = face_index == 3
    coordinates[0, this_face] = np.arctan(
        -unit_vectors.x[this_face] / unit_vectors.y[this_face]
    )
    coordinates[1, this_face] = np.arctan(
        -unit_vectors.z[this_face] / unit_vectors.y[this_face]
    )

    this_face = face_index == 4
    coordinates[0, this_face] = np.arctan(
        unit_vectors.x[this_face] / unit_vectors.z[this_face]
    )
    coordinates[1, this_face] = np.arctan(
        unit_vectors.y[this_face] / unit_vectors.z[this_face]
    )

    this_face = face_index == 5
    coordinates[0, this_face] = np.arctan(
        -unit_vectors.x[this_face] / unit_vectors.z[this_face]
    )
    coordinates[1, this_face] = np.arctan(
        -unit_vectors.y[this_face] / unit_vectors.z[this_face]
    )

    return face_index, coordinates


def _smooth_gnom_cube_histograms(
    histograms: np.ndarray[float],
    step_parameter: float,
    iterations: int = 1,
) -> np.ndarray[float]:
    """Histograms shape is (6, n_nbins, n_bins) and edge connectivity
    is as according to the rest of this file.
    """
    output_histogram = np.copy(histograms)
    diffused_weight = np.zeros(histograms.shape)

    for n in range(iterations):

        diffused_weight[...] = 0

        # Diffuse on faces
        for fi in range(6):
            diffused_weight[fi, 1:, :] += output_histogram[fi, :-1, :]
            diffused_weight[fi, :-1, :] += output_histogram[fi, 1:, :]
            diffused_weight[fi, :, 1:] += output_histogram[fi, :, :-1]
            diffused_weight[fi, :, :-1] += output_histogram[fi, :, 1:]

        connected_edge_pairs = (
            ((2, slice(None), -1), (4, slice(None), -1)),  # +y+z
            ((3, slice(None), -1), (4, slice(None), 0)),  # -y+z
            ((2, slice(None), 0), (5, slice(None), -1)),  # +y-z
            ((3, slice(None), 0), (5, slice(None), 0)),  # -y-z
            ((0, slice(None), -1), (4, -1, slice(None))),  # +x+z
            ((1, slice(None), -1), (4, 0, slice(None))),  # -x+z
            ((0, slice(None), 0), (5, -1, slice(None))),  # +x-z
            ((1, slice(None), 0), (5, 0, slice(None))),  # -x-z
            ((0, -1, slice(None)), (2, -1, slice(None))),  # +x+y
            ((1, -1, slice(None)), (2, 0, slice(None))),  # -x+y
            ((0, 0, slice(None)), (3, -1, slice(None))),  # +x-y
            ((1, 0, slice(None)), (3, 0, slice(None))),  # -x-y
        )

        for edge_1, edge_2 in connected_edge_pairs:
            diffused_weight[edge_1] += output_histogram[edge_2]
            diffused_weight[edge_2] += output_histogram[edge_1]

        # Add to output
        output_histogram = (
            1 - step_parameter
        ) * output_histogram + diffused_weight / 4 * step_parameter

    return output_histogram
