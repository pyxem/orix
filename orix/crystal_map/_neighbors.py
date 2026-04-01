import numpy as np

from orix.crystal_map import CrystalMap
from orix.quaternion import Orientation

from skimage.morphology._util import _raveled_offsets_and_distances
from skimage.util._map_array import map_array

def Neighbors(data,foot,mask=None):  
    """Performs the general preperations to do neighbor-based calculations on CrystalMaps in a vectorized form
        
    Args:
        data: 2d array. Can be a CrystalMap object
        mask: 2D array with same shape as CrystalMap indicating what points are valid or not eg. Non-indexed points
        foot: The kernel that defines which neighbors to look at. Can be both binary and not
        
    Returns: (maybe should be returned as an object instead?)
        indices: Array where the index for each valid point is repeated as many times as it has valid neighbors
        neighbor_indices: Array with the corosponding valid neighbors
        foot_values: contains how each neighbor in neighbor_indices is weighted defined by a given footprine/kernel
        num_neigbors_valid: the number of neighbors each valid point has
        nodes_valid: The index for valid points with at least one valid neighbor
        nodes_no_valid_neighbors: The index for valid points with no valid neighbors
        nodes_not_valid: The index for non-valid points 
        
    Raises:
        Have to add ValueErrors
    """
    if mask is None:
        if isinstance(data, CrystalMap):
            mask = data.is_indexed.reshape(data.shape)
        else:
            mask = np.full(data.shape, True) #All pointd are valid
    nodes = np.flatnonzero(mask)
    
    pad = np.max(foot.shape)//2
    padded_mask = np.pad(mask, pad, mode='constant', constant_values=False) 
    padded_nodes = np.flatnonzero(padded_mask)

    #The offset are given in 1D for the neighbors defined in 2D
    neighbor_offsets, dist = _raveled_offsets_and_distances(padded_mask.shape, footprint=foot)

    padded_neighbors = padded_nodes[:, np.newaxis] + neighbor_offsets
    neighbors = map_array(padded_neighbors, padded_nodes, nodes) #finds the indeces (in a non-padded format) for the neighbors belonging to valid points
    neighbors_mask = padded_mask.reshape(-1)[padded_neighbors] #sets which of those neigbors are valid
    
    num_neighbors = np.sum(neighbors_mask, axis=1)#the number of neighbors each valid point has
    indices = np.repeat(nodes, num_neighbors) #Array where the index for each valid point is repeated as many times as it has valid neighbors
    neighbor_indices = neighbors[neighbors_mask] #Array with the corosponding valid neighbors

    #Support for non-binary footprints 
    foot_offsets, dist = _raveled_offsets_and_distances(foot.shape, footprint=foot)
    foot_nodes = foot_offsets+foot.size//2
    foot_values = foot.reshape(-1)[np.repeat([foot_nodes],mask.sum(),axis=0)[neighbors_mask]]#contains how each neighbor in neighbor_indices is weighted defined by a given footprine

    valid_neighbor_mask = num_neighbors !=0
    nodes_valid = nodes[valid_neighbor_mask] #The index for valid points with at least one valid neighbor
    nodes_no_valid_neighbors = nodes[~valid_neighbor_mask] #The index for valid points with no valid neighbors
    nodes_invalid = np.flatnonzero(~mask) #The index for non-valid points (same as the mask input, not really neccesary)
    
    num_neighbors_valid = num_neighbors[valid_neighbor_mask]
    
    return(indices, neighbor_indices, foot_values, num_neighbors_valid, nodes_valid, nodes_no_valid_neighbors, nodes_invalid)


def neighbor_misorientation(xmap, indices, neighbor_indices, degree=False, crystall_symmetry=None):
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
        crystall_symmetry = xmap.phases[0].point_group #This can pehaps lead to problems, maybe make a check for multiple phases
    
    O_central_points = Orientation(xmap.rotations[indices], crystall_symmetry)
    O_neighbors = Orientation(xmap.rotations[neighbor_indices], crystall_symmetry)
    mis_ori = O_central_points.angle_with(O_neighbors)

    if degree:
        mis_ori = mis_ori*180/np.pi
    
    return mis_ori

def KAM_calc(xmap, mis_ori, non_index_value, no_neighbors_value, num_neighbors_valid, nodes_valid, nodes_no_valid_neighbors, nodes_invalid, foot_values=None):
    
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
        cumulative_miso = np.add.reduceat(mis_ori, np.r_[0, np.cumsum(num_neighbors_valid)[:-1]]) 
    else:
        cumulative_miso = np.add.reduceat(mis_ori*foot_values, np.r_[0, np.cumsum(num_neighbors_valid)[:-1]])
    
    kam_map = np.full(xmap.size, np.nan, dtype=np.float32)
    kam_map[nodes_valid] = cumulative_miso/num_neighbors_valid
    kam_map[nodes_no_valid_neighbors] = no_neighbors_value
    kam_map[nodes_invalid] = non_index_value
    kam_map_im = kam_map.reshape(xmap.shape)
    
    return kam_map_im

def NSN_calc(xmap, mis_ori, non_index_value, no_neighbors_value, lim, num_neighbors_valid, nodes_valid, nodes_no_valid_neighbors, nodes_invalid, foot_values=None):
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
        NSN = np.add.reduceat(same_neighbors, np.r_[0, np.cumsum(num_neighbors_valid)[:-1]]) 
    else:
        NSN = np.add.reduceat(same_neighbors*foot_values, np.r_[0, np.cumsum(num_neighbors_valid)[:-1]])
    
    NSN_map = np.full(xmap.size, np.nan, dtype=np.float32)
    NSN_map[nodes_valid] = NSN 
    NSN_map[nodes_no_valid_neighbors] = no_neighbors_value
    NSN_map[nodes_invalid] = non_index_value
        
    NSN_map_im = NSN_map.reshape(xmap.shape)
    
    return NSN_map_im