import numpy as np
from scipy.spatial import cKDTree


def _get_start_and_end_index(
    number_of_steps: float,
    include_start: bool,
    include_end: bool,
    positive_and_negative: bool,
):
    if positive_and_negative:
        start = -number_of_steps
    else:
        start = 0
    if not include_start:
        start = start + 1
    end = number_of_steps
    if include_end:
        end = end + 1
    return start, end


def _number_of_equidistant_steps(resolution: float, length: float) -> int:
    maximum_grid_spacing = np.tan(np.radians(resolution))
    return int(np.ceil(length / maximum_grid_spacing))


def _sample_length_equidistant(
    number_of_steps: int,
    length: float,
    include_start: bool = True,
    include_end: bool = False,
    positive_and_negative: bool = True,
):
    start_index, end_index = _get_start_and_end_index(
        number_of_steps,
        include_start,
        include_end,
        positive_and_negative,
    )
    grid_spacing = length / number_of_steps
    grid_on_edge = np.arange(start_index, end_index) * grid_spacing
    return grid_on_edge


def _number_of_equiangular_steps(resolution: float, length: float) -> int:
    total_angle = np.arctan(length)
    return int(np.ceil(total_angle / np.radians(resolution)))


def _sample_length_equiangular(
    number_of_steps: float,
    length: float,
    include_start: bool = True,
    include_end: bool = False,
    positive_and_negative: bool = True,
):
    total_angle = np.arctan(length)
    start_index, end_index = _get_start_and_end_index(
        number_of_steps,
        include_start,
        include_end,
        positive_and_negative,
    )
    linear_grid = np.arange(start_index, end_index)
    angular_increment = total_angle / number_of_steps
    grid_on_edge = np.tan(linear_grid * angular_increment)
    return grid_on_edge


def _edge_grid_normalized_cube(resolution: float):
    center_of_face_to_edge = 1.0
    number_of_steps = _number_of_equidistant_steps(resolution, center_of_face_to_edge)
    return _sample_length_equidistant(number_of_steps, center_of_face_to_edge)


def _edge_grid_spherified_edge_cube(resolution: float):
    center_of_face_to_edge = 1.0
    number_of_steps = _number_of_equiangular_steps(resolution, center_of_face_to_edge)
    return _sample_length_equiangular(number_of_steps, center_of_face_to_edge)


def _edge_grid_spherified_corner_cube(resolution: float):
    center_of_face_to_corner = np.sqrt(2)
    number_of_steps = _number_of_equiangular_steps(resolution, center_of_face_to_corner)
    grid_on_diagonal = _sample_length_equiangular(
        number_of_steps, center_of_face_to_corner
    )
    return grid_on_diagonal / center_of_face_to_corner


def _compose_from_faces(corners, faces, n):
    """
    Helper function to refine a grid starting from a platonic solid,
    adapted from meshzoo

    Parameters
    ----------
    corners: numpy.ndarray (N, 3)
        Coordinates of vertices for starting shape
    faces : list of 3-tuples of int elements
        Each tuple in the list corresponds to the vertex indices making
        up the face of the mesh
    n : int
        number of times the mesh is refined

    Returns
    -------
    vertices: numpy.ndarray (N, 3)
        The coordinates of the refined mesh vertices.

    See also
    --------
    :func:`get_icosahedral_mesh_vertices`
    """
    # create corner nodes
    vertices = [corners]
    vertex_count = len(corners)
    corner_nodes = np.arange(len(corners))
    # create edges
    edges = set()
    for face in faces:
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))
    edges = list(edges)
    # create edge nodes:
    edge_nodes = {}
    t = np.linspace(1 / n, 1.0, n - 1, endpoint=False)
    corners = vertices[0]
    k = corners.shape[0]
    for edge in edges:
        i0, i1 = edge
        new_vertices = np.outer(1 - t, corners[i0]) + np.outer(t, corners[i1])
        vertices.append(new_vertices)
        vertex_count += len(vertices[-1])
        edge_nodes[edge] = np.arange(k, k + len(t))
        k += len(t)
    triangle_cells = []
    k = 0
    for i in range(n):
        j = np.arange(n - i)
        triangle_cells.append(np.column_stack([k + j, k + j + 1, k + n - i + j + 1]))
        j = j[:-1]
        triangle_cells.append(
            np.column_stack([k + j + 1, k + n - i + j + 2, k + n - i + j + 1])
        )
        k += n - i + 1
    triangle_cells = np.vstack(triangle_cells)
    for face in faces:
        corners = face
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        is_edge_reverted = [False, False, False]
        for k, edge in enumerate(edges):
            if edge[0] > edge[1]:
                edges[k] = (edge[1], edge[0])
                is_edge_reverted[k] = True
        # First create the interior points in barycentric coordinates
        if n == 1:
            num_new_vertices = 0
        else:
            bary = (
                np.hstack(
                    [[np.full(n - i - 1, i), np.arange(1, n - i)] for i in range(1, n)]
                )
                / n
            )
            bary = np.array([1.0 - bary[0] - bary[1], bary[1], bary[0]])
            corner_verts = np.array([vertices[0][i] for i in corners])
            vertices_cart = np.dot(corner_verts.T, bary).T

            vertices.append(vertices_cart)
            num_new_vertices = len(vertices[-1])
        # translation table
        num_nodes_per_triangle = (n + 1) * (n + 2) // 2
        tt = np.empty(num_nodes_per_triangle, dtype=int)
        # first the corners
        tt[0] = corner_nodes[corners[0]]
        tt[n] = corner_nodes[corners[1]]
        tt[num_nodes_per_triangle - 1] = corner_nodes[corners[2]]
        # then the edges.
        # edge 0
        tt[1:n] = edge_nodes[edges[0]]
        if is_edge_reverted[0]:
            tt[1:n] = tt[1:n][::-1]
        #
        # edge 1
        idx = 2 * n
        for k in range(n - 1):
            if is_edge_reverted[1]:
                tt[idx] = edge_nodes[edges[1]][n - 2 - k]
            else:
                tt[idx] = edge_nodes[edges[1]][k]
            idx += n - k - 1
        #
        # edge 2
        idx = n + 1
        for k in range(n - 1):
            if is_edge_reverted[2]:
                tt[idx] = edge_nodes[edges[2]][k]
            else:
                tt[idx] = edge_nodes[edges[2]][n - 2 - k]
            idx += n - k
        # now the remaining interior nodes
        idx = n + 2
        j = vertex_count
        for k in range(n - 2):
            for _ in range(n - k - 2):
                tt[idx] = j
                j += 1
                idx += 1
            idx += 2
        vertex_count += num_new_vertices
    vertices = np.concatenate(vertices)
    return vertices


def _get_first_nearest_neighbors(points, leaf_size=50):
    """
    Helper function to get an array of first nearest neighbor points
    for all points in a point cloud

    Parameters
    ----------
    points : numpy.ndarray (N, D)
        Point cloud with N points in D dimensions
    leaf_size : int
        The NN search is performed using a cKDTree object. The way
        this tree is constructed depends on leaf_size, so this parameter
        will influence speed of tree construction and search.

    Returns
    -------
    nn1_vec : numpy.ndarray (N,D)
        Point cloud with N points in D dimensions, representing the nearest
        neighbor point of each point in "points"
    """
    tree = cKDTree(points, leaf_size)
    # get the indexes of the first nearest neighbor of each vertex
    nn1 = tree.query(points, k=2)[1][:, 1]
    nn1_vec = points[nn1]
    return nn1_vec


def _get_angles_between_nn_gridpoints(vertices, leaf_size=50):
    """
    Helper function to get the angles between all nearest neighbor grid
    points on a grid of a sphere.
    """
    # normalize the vertex vectors
    vertices = (vertices.T / np.linalg.norm(vertices, axis=1)).T
    nn1_vec = _get_first_nearest_neighbors(vertices, leaf_size)
    # the dot product between each point and its nearest neighbor
    nn_dot = np.sum(vertices * nn1_vec, axis=1)
    # angles
    angles = np.rad2deg(np.arccos(nn_dot))
    return angles


def _get_max_grid_angle(vertices, leaf_size=50):
    """
    Helper function to get the maximum angle between nearest neighbor grid
    points on a grid.
    """
    return np.max(_get_angles_between_nn_gridpoints(vertices, leaf_size))
