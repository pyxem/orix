#
# Copyright 2018-2026 the orix developers
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

from orix.crystal_map import CrystalMap
from orix.quaternion import Orientation


def _raveled_offsets(array_shape: tuple, kernel: np.ndarray):
    """Compute indicies offsets for neighboring pixels.

    This function is roughly equivalent to

        ```_raveled_offsets_and_distances(array_shape,footprint)[0]```

    in scikit-image.morphology._util, version 0.26.0. It calculates
    the relative indicies of an arbitrary pixel's neightbors, which
    can be used to quickly calculate all possible pixel-pixel
    neighbor pairs for any subset of points in an n-dimensional grid.

    Parameters
    ----------
    array_shape
        the shape of the array the kernel is being applied to.
        equivalent to `array.shape`.

    kernel
        the N-dimensional array representing the neighborhood, where
        N is the length of `array_shape`. This array is often a 2D
        Von Neumann neighborhood such as [[0,1,0],[1,0,1],[0,1,0]],
        but could also be a larger array of weighted values. All
        non-zero entries will produce an offset value.

    Returns
    -------
    raveled_offsets
        an array of offsets for each non-zero entry in the kernel.
    """
    # Developer note: this function is written to work for
    # arrays of any dimension, includeing 3D or 4D crystal maps.
    center = tuple(s // 2 for s in kernel.shape)
    offsets = np.stack(
        [(idx - c) for idx, c in zip(np.nonzero(kernel), center)], axis=-1
    )
    ravel_factors = array_shape[1:] + (1,)
    raveled_offsets = (offsets * ravel_factors).sum(axis=1)
    return np.sort(raveled_offsets)


def _find_neighbors(
    feature_map: np.ndarray,
    kernel: int | np.ndarray = 8,
):
    """Return indicies of alike neighbors for each pixel in a 2D map.

    The value of every pixel is compared to every other pixel in it's
    local neighborhood, as defined by the 'kernel' variable.
    The indicies of neighbors with matching values are then
    returned.

    For example, If feature_map is a boolean mask of indexed pixels,
    this will return all pixel-pixel conections in the indexed data.
    If feature_map is an array of grain IDs, this will return
    all pixel-pixel connections within a grain. In either case, pixels
    with a value of zero will be ignored.

    Parameters
    ----------
    feature_map
        A 2D numpy array.
    kernel
        Either an array describing a per-pixel neighborhood, or '4'
        or `8` to indicate the Von Neumman Neighborhoods of size 4
        and 8 respectively.

    Returns
    -------
    feature_idxs
        the indicies of the non-zero values in feature_map
    neighbors
        An n-by-m array of indices, where n is the number of non-zero
        pixels in `feature_map`, and m is the number of neighbors
        defined by 'kernel'. The values refer to each neighbor's
        relative position in a flattened version of the feature_map,
        with invalid connections replaced by the value -1.
    """
    # Note to future Devs: This currently supports only 2D grids, but was
    # written with the Developer note: this method currently only supports 2D grids, but
    # was written with the intention of supporting 3D gridded and ungridded
    # data in the future.

    # Convert kernel to a 2D array of floats
    if isinstance(kernel, int):
        von_neuman_dict = {
            4: np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            8: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
        }
        try:
            kernel = von_neuman_dict[kernel]
        except KeyError:
            raise ValueError("'kernel` must be either 4, 8, or a 2D numpy array")
    kernel = np.atleast_2d(kernel).astype(float)

    # assert the mask and kernel are 2D and of the correct data type.
    feature_map = np.atleast_2d(feature_map)
    if not np.all([feature_map.ndim == 2, kernel.ndim == 2]):
        raise ValueError("_find_neighbors only supports 2D arrays")

    pad = [(max(x // 2, 1), max(x - (x // 2) - 1, 1)) for x in kernel.shape]
    # Neighbor lookup needs to happen on a padded array to avoid wraparound
    padded_map = np.pad(
        array=feature_map,
        pad_width=pad,
        mode="constant",
        constant_values=0,
    )
    feature_idxs = np.flatnonzero(feature_map)
    padded_idxs = np.flatnonzero(padded_map)
    offsets = _raveled_offsets(padded_map.shape, kernel)
    neighbors = padded_idxs[:, np.newaxis] + offsets
    # NOTE: this section is roughly equivalent to the following, faster
    # Cython code:
    #
    #    from skimage.util._map_array import map_array
    #    neighbors = map_array(neighbors, padded_nodes, nodes+1) -1
    #
    # However, _map_array is a private function in skimage 0.26, so the
    # following vectorized method is used instead. If this slowdown becomes
    # problematic in the future, we should consider writing our own cython
    # code.
    is_neighbor = np.isin(neighbors, padded_idxs)
    depad_dict = dict(zip(padded_idxs, feature_idxs))
    depad_func = np.vectorize(lambda x: depad_dict.get(x, -1))
    neighbors[is_neighbor] = depad_func(neighbors[is_neighbor])

    neighbors[~is_neighbor] = -1
    return feature_idxs, neighbors


def kernel_average_misorientation_map(
    feature_map: np.ndarray,
    oris: Orientation,
    kernel: int | np.ndarray = 8,
):
    """
    Returns a Kernel-Averaged Misorientation (KAM) map.

    For each non-zero pixel in feature_map, the misorientation angle
    is calculated between it and each



    and each pixel with the same feature_map value in it's local
    neighborhood, as defined by the 'kernel'. The average angle
    is then returned per-pixel. pixels with no neighbors are assigned
    a KAM value of zero.

    Parameters
    ----------
    xtal_map
        Either a 1D or 2D
        A 2D CrystalMap, Orientation, or Rotation object.
    kernel
        Either an array describing a per-pixel neighborhood, or '4'
        or `8` to indicate the Von Neumman Neighborhoods of size 4
        and 8 respectively.
    feature_map
        A 2D numpy array of integers or booleans. Must have the
        same dimensions as `xtal_map`. If not given, assume all
        pixels are potentially valid neighbors.

    Returns
    -------
    kam_map
        a 2D numpy array of kam angles.


    """
    map_idxs, map_neighbors = _find_neighbors(feature_map, kernel)
    n_count = np.sum(map_neighbors > -1, axis=1)
    l_idxs = np.repeat(map_idxs, n_count)
    r_idxs = map_neighbors[map_neighbors > -1]

    map2ori = np.zeros(map_idxs.max() + 2, dtype=int) - 1
    map2ori[map_idxs] = np.arange(len(map_idxs))
    m_angle = (oris[map2ori[l_idxs]] * ~oris[map2ori[r_idxs]]).angle
    kam_map = np.zeros(feature_map.size, dtype=float)
    if isinstance(kernel, np.ndarray):
        weights = kernel[kernel > 0]
        if len(np.uniuqe(weights)) > 1:
            m_angle = (
                m_angle
                * np.repeat(weights[np.newaxis, :], oris.size, axis=0)[
                    map_neighbors > -1
                ]
                / np.sum(weights)
            )
    kam_map[l_idxs] += m_angle
    kam_map[map_idxs[n_count > 0]] /= n_count[n_count > 0]
    kam_map = kam_map.reshape(feature_map.shape)

    return kam_map


def number_same_neighbors_map(
    feature_map: np.ndarray,
    oris: Orientation,
    kernel: int | np.ndarray = 8,
    cutoff_angle=5,
):
    map_idxs, map_neighbors = _find_neighbors(feature_map, kernel)
    n_count = np.sum(map_neighbors > -1, axis=1)
    l_idxs = np.repeat(map_idxs, n_count)
    r_idxs = map_neighbors[map_neighbors > -1]

    map2ori = np.zeros(map_idxs.max() + 2, dtype=int) - 1
    map2ori[map_idxs] = np.arange(len(map_idxs))
    m_angle = (oris[map2ori[l_idxs]] * ~oris[map2ori[r_idxs]]).angle
    nsn = m_angle < (cutoff_angle * np.pi / 180)

    nsn_map = np.zeros(feature_map.size, dtype=float)
    nsn_map[l_idxs] += nsn
    nsn_map = nsn_map.reshape(feature_map.shape)

    return nsn_map
