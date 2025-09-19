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
from scipy.ndimage import gaussian_filter

from orix.projections.stereographic import StereographicProjection
from orix.quaternion.symmetry import Symmetry
from orix.vector.vector3d import Vector3d
from scipy.interpolate import RegularGridInterpolator

def pole_density_function(
    *args: np.ndarray | Vector3d,
    resolution: float = 1,
    sigma: float = 5,
    weights: np.ndarray | None = None,
    hemisphere: str = "upper",
    symmetry: Symmetry | None = None,
    log: bool = False,
    mrd: bool = True,
) -> tuple[np.ma.MaskedArray, tuple[np.ndarray, np.ndarray]]:
    """Compute the Pole Density Function (PDF) of vectors in the
    stereographic projection. See :cite:`rohrer2004distribution`.

    If ``symmetry`` is defined then the PDF is folded back into the
    point group fundamental sector and accumulated.

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
    from orix.sampling.S2_sampling import _sample_S2_equal_area_coordinates

    hemisphere = hemisphere.lower()
    poles = {"upper": -1, "lower": 1}
    sp = StereographicProjection(poles[hemisphere])

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
            "Accepts only one (Vector3d) or two (azimuth, polar) input arguments."
        )

    if symmetry is not None:
        v = v.in_fundamental_sector(symmetry)
        # To help with aliasing after reprojection into point group
        # fundamental sector in the inverse pole figure case, the
        # initial sampling is performed at half the angular resolution
        resolution /= 2

    # azimuth, polar, _ = v.to_polar()
    face_index_array, face_coordinates = _cube_gnom_coordinates(v)
    # np.histogram2d expects 1d arrays
    face_index_array = face_index_array.ravel()
    face_coordinate_1 = face_coordinates[0].ravel()
    face_coordinate_2 = face_coordinates[1].ravel()

    bins = int(90 / resolution)
    bin_edges = np.linspace(-np.pi/4, np.pi/4, bins+1)
    hist = np.zeros((6, bins, bins))
    for face_index in range(6):
        this_face = face_index_array == face_index
        hist[face_index], _ = np.histogramdd(
                (face_coordinate_1[this_face], face_coordinate_2[this_face]),
                bins=(bin_edges, bin_edges),
            )

    # Bins are not all same solid angle area, so we need to normalize.
    if mrd:
        bin_middles = np.linspace(-np.pi/4 + np.pi/4/bins, np.pi/4 - np.pi/4/bins, bins)
        y_ang, z_ang = np.meshgrid(bin_middles, bin_middles)
        solid_angle_term = 1 / (np.tan(y_ang)**2 + np.tan(z_ang)**2 + 1)/\
            (np.cos(y_ang)*np.cos(z_ang)) / (1 - 0.5*(np.sin(z_ang) * np.sin(y_ang))**2)
        solid_angle_term *= 1 / 6 / np.sum(solid_angle_term)
        solid_angle_term *= v.size
        hist = hist / solid_angle_term[np.newaxis, ...]

    # TODO: If the plot resolution is very high, and the smoothing kernel is very broad
    # this blurring has bad performance. In those cases, overwrite the users choice
    # for resolution, and then upsample in the end.

    if sigma/resolution > 20 and resolution<1.0:
        print('Performance is bad when smoothing-kernel is much larger than\
         the resoltion. Consider using a lower resolution.')

    if sigma != 0.0:
        # If smoothing is only a bit, du 60 small steps, otherwise do a max step-size
        if (sigma / resolution)**2 <= 20:
            N = 60
            t = 1 / N * (sigma / resolution)**2
        else:
            t = 1 / 3
            N = int(1 / t * (sigma / resolution)**2)

        hist = _smooth_gnom_cube_histograms(hist, t, N)

    # TODO For now, avoid touching the plotting code, by returning the
    # expected format. I will have a deep dive in the plotting later.
    azimuth_coords, polar_coords = _sample_S2_equal_area_coordinates(
        1,
        hemisphere='upper',
        azimuth_endpoint=True,
    )
    azimuth_coords
    azimuth_grid, polar_grid = np.meshgrid(azimuth_coords, polar_coords, indexing="ij")

    azimuth_center_grid, polar_center_grid = np.meshgrid(
        azimuth_coords[:-1] + np.diff(azimuth_coords) / 2,
        polar_coords[:-1] + np.diff(polar_coords) / 2,
        indexing="ij",
    )
    v_center_grid = Vector3d.from_polar(
        azimuth=azimuth_center_grid, polar=polar_center_grid
    ).unit

    grid_face_index, grid_face_coords = _cube_gnom_coordinates(v_center_grid)
    hist_grid = np.zeros(v_center_grid.shape)
    bin_middles = np.linspace(-np.pi/4 + np.pi/4/bins, np.pi/4 - np.pi/4/bins, bins)

    for face_index in range(6):
        interpolator = RegularGridInterpolator((bin_middles, bin_middles), hist[face_index], bounds_error=False, fill_value=None)
        this_face = grid_face_index == face_index
        hist_grid[this_face] = interpolator(grid_face_coords[:, this_face].T)

    hist = hist_grid

    # TODO: I don't understand how symmetrization is done. Could just sum over
    # rotated histograms instead.

    # In the case of inverse pole figure, accumulate all values outside
    # of the point group fundamental sector back into correct bin within
    # fundamental sector

    if symmetry is not None:
        # compute histogram bin centers in azimuth and polar coords
        azimuth_center_grid, polar_center_grid = np.meshgrid(
            azimuth_coords[:-1] + np.diff(azimuth_coords) / 2,
            polar_coords[:-1] + np.diff(polar_coords) / 2,
            indexing="ij",
        )
        v_center_grid = Vector3d.from_polar(
            azimuth=azimuth_center_grid, polar=polar_center_grid
        ).unit
        # fold back in into fundamental sector
        v_center_grid_fs = v_center_grid.in_fundamental_sector(symmetry)
        azimuth_center_fs, polar_center_fs, _ = v_center_grid_fs.to_polar()
        azimuth_center_fs = azimuth_center_fs.ravel()
        polar_center_fs = polar_center_fs.ravel()

        # Generate coorinates with user-defined resolution.
        # When `symmetry` is defined, the initial grid was calculated
        # with `resolution = resolution / 2`
        azimuth_coords_res2, polar_coords_res2 = _sample_S2_equal_area_coordinates(
            2 * resolution,
            hemisphere=hemisphere,
            azimuth_endpoint=True,
        )
        azimuth_res2_grid, polar_res2_grid = np.meshgrid(
            azimuth_coords_res2, polar_coords_res2, indexing="ij"
        )
        v_res2_grid = Vector3d.from_polar(
            azimuth=azimuth_res2_grid, polar=polar_res2_grid
        )

        # calculate histogram values for vectors folded back into
        # fundamental sector
        i = np.digitize(azimuth_center_fs, azimuth_coords_res2[1:-1])
        j = np.digitize(polar_center_fs, polar_coords_res2[1:-1])
        # recompute histogram
        temp = np.zeros((azimuth_coords_res2.size - 1, polar_coords_res2.size - 1))
        # add hist data to new histogram without buffering
        np.add.at(temp, (i, j), hist.ravel())

        # get new histogram bins centers to compute histogram mask
        azimuth_center_res2_grid, polar_center_res2_grid = np.meshgrid(
            azimuth_coords_res2[:-1] + np.ediff1d(azimuth_coords_res2) / 2,
            polar_coords_res2[:-1] + np.ediff1d(polar_coords_res2) / 2,
            indexing="ij",
        )
        v_center_res2_grid = Vector3d.from_polar(
            azimuth=azimuth_center_res2_grid, polar=polar_center_res2_grid
        ).unit

        # compute histogram data array as masked array
        hist = np.ma.array(
            temp, mask=~(v_center_res2_grid <= symmetry.fundamental_sector)
        )
        # calculate bin vertices
        x, y = sp.vector2xy(v_res2_grid)
        x, y = x.reshape(v_res2_grid.shape), y.reshape(v_res2_grid.shape)

        # This was missing before
        hist = hist / symmetry.size

    else:
        # all points valid in stereographic projection
        hist = np.ma.array(hist, mask=np.zeros_like(hist, dtype=bool))
        # calculate bin vertices
        v_grid = Vector3d.from_polar(azimuth=azimuth_grid, polar=polar_grid).unit
        x, y = sp.vector2xy(v_grid)
        x, y = x.reshape(v_grid.shape), y.reshape(v_grid.shape)

    if log:
        # +1 to avoid taking the log of 0
        hist = np.log(hist + 1)

    return hist, (x, y)


def _cube_gnom_coordinates(vectors: Vector3d) -> tuple[np.ndarray[int], np.ndarray[float]]:
    """ Assigns an index (0 to 5) to an array of ```Vector3d```
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
    """

    if np.any(vectors.norm == 0.0):
        #TODO  Ask the maintainers what they normally do here.
        raise ZeroDivisionError

    # Assign face index to each vector
    face_index = np.zeros(vectors.shape, dtype=int)

    indx = np.all([vectors.x >= vectors.y,
                   vectors.x >= -vectors.y,
                   vectors.x >= vectors.z,
                   vectors.x >= -vectors.z,], axis=0)
    face_index[indx] = 0

    indx = np.all([vectors.x <= vectors.y,
                   vectors.x <= -vectors.y,
                   vectors.x <= vectors.z,
                   vectors.x <= -vectors.z,], axis=0)
    face_index[indx] = 1

    indx = np.all([vectors.y > vectors.x,
                   vectors.y >= -vectors.x,
                   vectors.y >= vectors.z,
                   vectors.y > -vectors.z,], axis = 0)
    face_index[indx] = 2

    indx = np.all([vectors.y < vectors.x,
                   vectors.y <= -vectors.x,
                   vectors.y <= vectors.z,
                   vectors.y < -vectors.z,], axis = 0)
    face_index[indx] = 3

    indx = np.all([vectors.z > vectors.x,
                   vectors.z >= -vectors.x,
                   vectors.z > vectors.y,
                   vectors.z >= -vectors.y,], axis = 0)
    face_index[indx] = 4

    indx = np.all([vectors.z < vectors.x,
                   vectors.z <= -vectors.x,
                   vectors.z < vectors.y,
                   vectors.z <= -vectors.y,], axis = 0)
    face_index[indx] = 5


    # Assign coordinates
    coordinates = np.zeros((2,) + vectors.shape)
    unit_vectors = vectors.unit

    #  Comment: no need for np.arctan2. We are sure that the denominator is non-zero
    #  so np.arctan should be faster.

    this_face = face_index == 0
    coordinates[0, this_face] =\
        np.arctan(unit_vectors.y[this_face] / unit_vectors.x[this_face])
    coordinates[1, this_face] =\
        np.arctan(unit_vectors.z[this_face] / unit_vectors.x[this_face])

    this_face = face_index == 1
    coordinates[0, this_face] =\
        np.arctan(-unit_vectors.y[this_face] / unit_vectors.x[this_face])
    coordinates[1, this_face] =\
        np.arctan(-unit_vectors.z[this_face] / unit_vectors.x[this_face])

    this_face = face_index == 2
    coordinates[0, this_face] =\
        np.arctan(unit_vectors.x[this_face] / unit_vectors.y[this_face])
    coordinates[1, this_face] =\
        np.arctan(unit_vectors.z[this_face] / unit_vectors.y[this_face])

    this_face = face_index == 3
    coordinates[0, this_face] =\
        np.arctan(-unit_vectors.x[this_face] / unit_vectors.y[this_face])
    coordinates[1, this_face] =\
        np.arctan(-unit_vectors.z[this_face] / unit_vectors.y[this_face])

    this_face = face_index == 4
    coordinates[0, this_face] =\
        np.arctan(unit_vectors.x[this_face] / unit_vectors.z[this_face])
    coordinates[1, this_face] =\
        np.arctan(unit_vectors.y[this_face] / unit_vectors.z[this_face])

    this_face = face_index == 5
    coordinates[0, this_face] =\
        np.arctan(-unit_vectors.x[this_face] / unit_vectors.z[this_face])
    coordinates[1, this_face] =\
        np.arctan(-unit_vectors.y[this_face] / unit_vectors.z[this_face])

    return face_index, coordinates


def _smooth_gnom_cube_histograms(histograms, step_parameter, iterations=1):
    """ Histograms shape is (6, n_nbins, n_bins) and edge connectivity
    is as according to the rest of this file.
    """
    sub_histogram_shape = histograms[0][0].shape
    output_histogram = np.copy(histograms)
    diffused_weight = np.zeros(histograms.shape)

    for n in range(iterations):

        diffused_weight[...] = 0

        # Diffuse on faces
        for face_index in range(6):

            diffused_weight[face_index, 1:, :] += output_histogram[face_index, :-1, :]
            diffused_weight[face_index, :-1, :] += output_histogram[face_index, 1:, :]
            diffused_weight[face_index, :, 1:] += output_histogram[face_index, :, :-1]
            diffused_weight[face_index, :, :-1] += output_histogram[face_index, :, 1:]


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
            diffused_weight[edge_1] +=output_histogram[edge_2]
            diffused_weight[edge_2] +=output_histogram[edge_1]

        # Add to output
        output_histogram = (1-step_parameter)*output_histogram\
            + diffused_weight/4*step_parameter

    return output_histogram
