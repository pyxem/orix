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

from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree


def _get_start_and_end_index(
    number_of_steps: int,
    include_start: bool,
    include_end: bool,
    positive_and_negative: bool,
) -> Tuple[int, int]:
    if positive_and_negative:
        start = -number_of_steps
    else:
        start = 0
    if not include_start:
        start += 1
    end = number_of_steps
    if include_end:
        end += 1
    return start, end


def _number_of_equidistant_steps(resolution: float, length: float) -> int:
    maximum_grid_spacing = np.tan(np.deg2rad(resolution))
    return int(np.ceil(length / maximum_grid_spacing))


def _sample_length_equidistant(
    number_of_steps: int,
    length: float,
    include_start: bool = True,
    include_end: bool = False,
    positive_and_negative: bool = True,
) -> np.ndarray:
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
    return int(np.ceil(total_angle / np.deg2rad(resolution)))


def _sample_length_equiangular(
    number_of_steps: int,
    length: float,
    include_start: bool = True,
    include_end: bool = False,
    positive_and_negative: bool = True,
) -> np.ndarray:
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


def _edge_grid_normalized_cube(resolution: float) -> np.ndarray:
    center_of_face_to_edge = 1.0
    number_of_steps = _number_of_equidistant_steps(resolution, center_of_face_to_edge)
    return _sample_length_equidistant(number_of_steps, center_of_face_to_edge)


def _edge_grid_spherified_edge_cube(resolution: float) -> np.ndarray:
    center_of_face_to_edge = 1.0
    number_of_steps = _number_of_equiangular_steps(resolution, center_of_face_to_edge)
    return _sample_length_equiangular(number_of_steps, center_of_face_to_edge)


def _edge_grid_spherified_corner_cube(resolution: float) -> np.ndarray:
    center_of_face_to_corner = np.sqrt(2)
    number_of_steps = _number_of_equiangular_steps(resolution, center_of_face_to_corner)
    grid_on_diagonal = _sample_length_equiangular(
        number_of_steps, center_of_face_to_corner
    )
    return grid_on_diagonal / center_of_face_to_corner


def _compose_from_faces(
    corners: np.ndarray,
    faces: List[Tuple[int, int, int]],
    n: int,
) -> np.ndarray:
    """Refine a grid starting from a platonic solid; adapted from
    :mod:`meshzoo` :cite:`meshzoo`.

    Parameters
    ----------
    corners
        Coordinates of vertices for starting shape. Shape of the array
        should be (N, 3).
    faces
        Each tuple in the list corresponds to the vertex indices making
        up a triangular face of the mesh.
    n
        Number of times the mesh is refined.

    Returns
    -------
    vertices
        The coordinates of the refined mesh vertices, an array of shape
        (N, 3).
    """
    # create corner nodes
    vertices = [corners]

    # create edges
    edges = set()
    for face in faces:
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))

    # create edge nodes:
    t = np.linspace(1 / n, 1.0, n - 1, endpoint=False)
    for edge in edges:
        i0, i1 = edge
        new_vertices = np.outer(1 - t, corners[i0]) + np.outer(t, corners[i1])
        vertices.append(new_vertices)

    for face in faces:
        face_corner_indices = face

        bary = (
            np.hstack(
                [[np.full(n - i - 1, i), np.arange(1, n - i)] for i in range(1, n)]
            )
            / n
        )
        bary = np.array([1.0 - bary[0] - bary[1], bary[1], bary[0]])
        corner_verts = np.array([corners[i] for i in face_corner_indices])
        vertices_cart = np.dot(corner_verts.T, bary).T

        vertices.append(vertices_cart)

    return np.concatenate(vertices)


def _get_first_nearest_neighbors(
    points: np.ndarray,
    leaf_size: int = 50,
) -> np.ndarray:
    """Get array of first nearest neighbor points for all points in a
    point cloud.

    Parameters
    ----------
    points
        Point cloud with shape (N, D) representing N points in D
        dimensions.
    leaf_size
        The NN search is performed using a cKDTree object. The way this
        tree is constructed depends on ``leaf_size``, so this parameter
        will influence speed of tree construction and search.

    Returns
    -------
    nn1_vec
        Point cloud represented by an array of shape (N, D) with N
        points in D dimensions, representing the nearest neighbor point
        of each point in ``points``.
    """
    tree = cKDTree(points, leaf_size)
    # get the indexes of the first nearest neighbor of each vertex
    nn1 = tree.query(points, k=2)[1][:, 1]
    nn1_vec = points[nn1]
    return nn1_vec


def _get_angles_between_nn_gridpoints(
    vertices: np.ndarray,
    leaf_size: int = 50,
) -> np.ndarray:
    """Return angles between all nearest neighbor grid points on unit
    sphere.
    """
    # normalize the vertex vectors
    vertices = (vertices.T / np.linalg.norm(vertices, axis=1)).T
    nn1_vec = _get_first_nearest_neighbors(vertices, leaf_size)
    # the dot product between each point and its nearest neighbor
    nn_dot = np.sum(vertices * nn1_vec, axis=1)
    # angles
    angles = np.rad2deg(np.arccos(nn_dot))
    return angles


def _get_max_grid_angle(
    vertices: np.ndarray,
    leaf_size: int = 50,
) -> np.ndarray:
    """Get the maximum angle between nearest neighbor grid points on a
    unit sphere.
    """
    return np.max(_get_angles_between_nn_gridpoints(vertices, leaf_size))
