import numpy as np
from skimage.morphology._util import _raveled_offsets_and_distances
from skimage.util._map_array import map_array

from orix.crystal_map import CrystalMap
from orix.quaternion import Orientation

def _raveled_offsets(array_shape: tuple,
                     kernel: np.ndarray):
    """Compute neighboring pixel offsets in raveled coordinate space.
    
    This is a simlification of the "_raveled_offsets_and_distances"
    function in scikit-image.morphology._util, version 0.26.0. 
    
    Parameters
    ----------    
    array_shape
        the shape of the array the kernel is being applied to.
        equivalent to `array.shape`.

    kernel
        the 2D array representing the neighborhood. this is normally
        a 2D Von Neumann neighborhood such as [[0,1,0],[1,0,1],[0,1,0]],
        but could also be a larger array of weighted values. all
        non-zero entries will produce an offset value.

    Returns
    -------
    raveled_offsets
        an array of offsets for each non-zero entry in the kernel.
    """
    center = tuple(s // 2 for s in kernel.shape)
    offsets = np.stack(
        [(idx - c) for idx, c in zip(np.nonzero(kernel), center)], axis=-1
    )
    ravel_factors = array_shape[1:] + (1,)
    raveled_offsets = (offsets * ravel_factors).sum(axis=1)
    return np.sort(raveled_offsets)

def _find_neighbors(mask: np.ndarray | CrystalMap, 
                    kernel: int | np.ndarray = 8,
                    return_indices: bool=False,
                    ):
    """Given a 2D boolean array, return the neighbors for each pixel.

    Parameters
    ---------- 
    mask
    kernel
    return_indices

    Returns
    -------
    neighbors
    indices
    """

    # Convert mask to a 2D aray of booleans
    if isinstance(mask, CrystalMap):
        mask = mask.is_indexed.reshape(mask.shape)
    mask = np.atleast_2d(mask).astype(bool)        
    # Convert kernel to a 2D array of floats
    if isinstance(kernel, int):
        von_neuman_dict = {
            4:np.array([[0,1,0],[1,0,1],[0,1,0]]),
            8:np.array([[1,1,1],[1,0,1],[1,1,1]]),
            }
        try:
            kernel = von_neuman_dict[kernel]
        except KeyError:
            raise ValueError(
                "'kernel` must be either 4, 8, or a 2D numpy array")
        kernel = np.atleast_2d(kernel).astype(float)
        if not np.all([mask.ndim ==2,kernel.ndim==2]):
            raise ValueError("_find_neighbors only supports 2D arrays")

    # calculate the padding, offsets, and indices's of the queried points. 
    # note: padding must be at least one in all directions.
    pad = [(max(x//2,1),max(x-(x//2)-1,1)) for x in kernel.shape]
    offsets = _raveled_offsets(padded_mask.shape, kernel)
    nodes = np.flatnonzero(mask)

    # calculate neighbors for the padded array.
    padded_mask = np.pad(mask, pad, mode='constant', constant_values=False) 
    padded_nodes = np.flatnonzero(padded_mask)
    neighbors = padded_nodes[:, np.newaxis] + offsets

    # replace the padded indices with the correct unpadded ones, and set
    # out-of-bounds or invalid neighbor indices to -1
    # NOTE: this section is roughly equivalent to the following, faster
    # Cython code:
    #    from skimage.util._map_array import map_array
    #    neighbors = map_array(neighbors, padded_nodes, nodes+1) -1
    #
    # However, _map_array is a private function in skimage 0.26, so the
    # following vectorized method is used instead. If this slowdown becomes
    # problematic in the future, we should consider writing our own cython
    # code.
    is_neighbor = np.isin(neighbors,padded_nodes)
    depad_dict = dict(zip(padded_nodes, nodes))
    depad_func = np.vectorize(lambda x: depad_dict.get(x,-1))
    neighbors[is_neighbor] = depad_func(neighbors[is_neighbor])
    neighbors[~is_neighbor] = -1
    
    if return_indices:
        return neighbors, indices
    
    return neigbors


def neighbor_misorientation(
    xmap, indices, neighbor_indices, degree=False, crystall_symmetry=None
):
    """Calculates the misorientation angles between a central point and its neighbors
    Args:
        xmap: CrystalMap object
        indices: Array where the index for each valid point is repeated as many times as it has valid neighbors
        neighbor_indices: Array with the corosponding valid neighbors
        degree (bool): True-return results in degree. False-return results in radiens
        crystall_symmetry: used to differentiate between rotation and orientation
    Returns:
        d: misorientation angles given in radiens

    Raises:
        Have to add ValueErrors
    """
    if crystall_symmetry is None:
        crystall_symmetry = xmap.phases[
            0
        ].point_group  # This can pehaps lead to problems, maybe make a check for multiple phases

    O_central_points = Orientation(xmap.rotations[indices], crystall_symmetry)
    O_neighbors = Orientation(xmap.rotations[neighbor_indices], crystall_symmetry)
    mis_ori = O_central_points.angle_with(O_neighbors)

    if degree:
        mis_ori = mis_ori * 180 / np.pi

    return mis_ori


def KAM_calc(
    xmap,
    mis_ori,
    non_index_value,
    no_neighbors_value,
    num_neighbors_valid,
    nodes_valid,
    nodes_no_valid_neighbors,
    nodes_invalid,
    foot_values=None,
):
    """Makes the KAM map
    Args:
        xmap: CrystalMap object (or any 2d array actually)
        mis_ori: Misorientation angles given in radiens or degrees (from neighbor_misorientation())
        non_index_value (int or float): Kam value given to a non-indexed points in the xmap
        no_neigbors_value (int or float): Kam value given to an indexed point without a single indexed neighbor

        These are from Neighbors()
        num_neigbors_valid: the number of neighbors each valid point has
        nodes_valid: The index for valid points with at least one valid neighbor
        nodes_no_valid_neighbors: The index for valid points with no valid neighbors
        nodes_not_valid: The index for non-valid points
        foot_values: The weight of different neighbors if non-binary foot is used

    Returns:
        kam_map_im: 2D array with same shape as xmap with all the KAM values

    Raises:
        Have to add ValueErrors
    """
    if foot_values is None:
        cumulative_miso = np.add.reduceat(
            mis_ori, np.r_[0, np.cumsum(num_neighbors_valid)[:-1]]
        )
    else:
        cumulative_miso = np.add.reduceat(
            mis_ori * foot_values, np.r_[0, np.cumsum(num_neighbors_valid)[:-1]]
        )

    kam_map = np.full(xmap.size, np.nan, dtype=np.float32)
    kam_map[nodes_valid] = cumulative_miso / num_neighbors_valid
    kam_map[nodes_no_valid_neighbors] = no_neighbors_value
    kam_map[nodes_invalid] = non_index_value
    kam_map_im = kam_map.reshape(xmap.shape)

    return kam_map_im


def NSN_calc(
    xmap,
    mis_ori,
    non_index_value,
    no_neighbors_value,
    lim,
    num_neighbors_valid,
    nodes_valid,
    nodes_no_valid_neighbors,
    nodes_invalid,
    foot_values=None,
):
    """Makes a Number of same neighbors (NSN) map. Each point in the xmap gets assigned the value equal to the number similar
       oriented neighbors. Where similar is defined by a user set limit
    Args:
        xmap: CrystalMap object (or any 2d array actually)
        mis_ori: Misorientation angles given in radiens or degrees (from neighbor_misorientation())
        non_index_value (int or float): NSN value given to a non-indexed points in the xmap
        no_neigbors_value (int or float): NSN value given to an indexed point without a single indexed neighbor
        lim (int or float): The limit for when two neighboring point are defined to have different/same orientations

        These are from Neighbors()
        num_neigbors_valid: the number of neighbors each valid point has
        nodes_valid: The index for valid points with at least one valid neighbor
        nodes_no_valid_neighbors: The index for valid points with no valid neighbors
        nodes_not_valid: The index for non-valid points
        foot_values: The weight of different neighbors if non-binary foot is used

    Returns:
        NDN_map_im: 2D array with same shape as xmap with all the NDN values

    Raises:
        Have to add ValueErrors
    """

    same_neighbors = mis_ori < lim
    if foot_values is None:
        NSN = np.add.reduceat(
            same_neighbors, np.r_[0, np.cumsum(num_neighbors_valid)[:-1]]
        )
    else:
        NSN = np.add.reduceat(
            same_neighbors * foot_values, np.r_[0, np.cumsum(num_neighbors_valid)[:-1]]
        )

    NSN_map = np.full(xmap.size, np.nan, dtype=np.float32)
    NSN_map[nodes_valid] = NSN
    NSN_map[nodes_no_valid_neighbors] = no_neighbors_value
    NSN_map[nodes_invalid] = non_index_value

    NSN_map_im = NSN_map.reshape(xmap.shape)

    return NSN_map_im
