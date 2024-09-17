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

import numpy as np

from orix.vector import SphericalRegion, Vector3d

_EDGE_STEPS = 1000


class FundamentalSector(SphericalRegion):
    """Fundamental sector for a symmetry in the inverse pole figure,
    defined by a set of sector normals.
    """

    # This is only set for T (23), Th (m-3) and O (432), in the
    # Symmetry.fundamental_sector property, because the UV S2 sampling
    # isn't uniform enough to produce the correct center according to
    # MTEX
    _center = None

    # Used when sorting `edges` for restricting stereographic plot
    _pole = -1

    # -------------------------- Properties -------------------------- #

    @property
    def vertices(self) -> Vector3d:
        """Return the sector vertices."""
        n = self.size
        if n == 0:
            return Vector3d.empty()
        else:
            normals = self.reshape(1, n)
            u = normals.cross(normals.transpose())
            return Vector3d(u[u <= self]).unique().unit

    @property
    def center(self) -> Vector3d:
        """Return the center vector of the fundamental sector.

        Taken from MTEX' :code:`sphericalRegion.center`.
        """
        v = self.vertices.unique()
        n_vertices = v.size
        n_normals = self.size
        if n_normals < 2:
            center = self
        elif n_vertices < 3:
            # Find the pair of maximum angle
            angles = self.angle_with(self.reshape(n_normals, 1))
            indices = np.argmax(angles, axis=1)
            center = self[indices].mean()
        elif n_vertices < 4:
            center = v.mean()
        elif isinstance(self._center, Vector3d):
            # Only the case for T (23), Th (m-3) and O (432), for which
            # the S2 sampling isn't uniform enough to produce the
            # correct center according to MTEX
            center = self._center
        else:
            # Avoid circular import
            from orix.sampling import sample_S2

            v_all = sample_S2(resolution=1, method="spherified_cube_corner")
            v = v_all[v_all < self]
            center = v.mean()

        return Vector3d(center)

    @property
    def edges(self) -> Vector3d:
        """Return the unit vectors which delineate the region in the
        stereographic projection.

        The vectors are sorted in the counter-clockwise direction
        around the sector center in the stereographic projection.

        The first edge is repeated at the end. This is done so that
        :meth:`orix.plot.StereographicPlot.plot` draws bounding lines
        without gaps.
        """
        if self.size == 0:
            return Vector3d.empty()

        circles = self.get_circle(steps=_EDGE_STEPS)
        edges = np.zeros((self.size * _EDGE_STEPS + 3, 3))
        vertices = self.vertices

        if vertices.size == 0:
            return circles.squeeze()

        j = 0
        for ci, vi in zip(circles, vertices):
            # Only get the parts of the great circle that are within
            # this spherical region
            ci_inside = ci[ci <= self]
            v_keep = Vector3d(np.vstack((ci_inside.data, vi.data)))
            v_keep = v_keep.unique()
            order = np.lexsort((v_keep.azimuth, v_keep.polar))
            v_keep = v_keep[order]
            v_n = v_keep.size
            edges[j : j + v_n] = v_keep.data
            j += v_n
        edges = Vector3d(edges[:j])

        order = _order_to_sort_around_center(edges, self.center, self._pole)
        sorted_edges = edges[order]

        return sorted_edges.squeeze()


def _order_to_sort_around_center(
    v: Vector3d, center: Vector3d, pole: int = -1
) -> np.ndarray:
    vz = Vector3d.zvector()
    if pole == 1:
        vz = -vz
    angle = vz.angle_with(center)
    axis = vz.cross(center)
    v_rotated = v.rotate(axis=axis, angle=-angle)

    order1 = np.argsort(v_rotated.azimuth)
    idx_closest_to_001 = np.argmax(v[order1].dot(vz))
    order2 = np.roll(order1, shift=-idx_closest_to_001)

    return order2


def _closed_edges_in_hemisphere(
    edges: Vector3d, sector: FundamentalSector, pole: int = -1
) -> Vector3d:
    if pole == -1:
        is_outside = edges.polar >= np.pi / 2
    else:  # pole == 1
        is_outside = edges.polar <= np.pi / 2

    if not np.any(is_outside):
        return edges
    elif np.all(is_outside):
        return Vector3d.empty()
    else:
        idx_after_crossing_equator = np.where(is_outside != is_outside[0])[0][0]
        equator = Vector3d.zvector().get_circle(steps=_EDGE_STEPS)
        equator_inside = equator[equator <= sector]
        edges_inside = edges[~is_outside]
        azimuth_before_equator = edges_inside[
            [idx_after_crossing_equator - 1, idx_after_crossing_equator]
        ].azimuth
        v_before_equator = Vector3d.from_polar(
            azimuth_before_equator, polar=[np.pi / 2] * 2
        )
        return Vector3d(
            np.vstack(
                (
                    edges_inside[:idx_after_crossing_equator].data,
                    v_before_equator[0].data,
                    equator_inside.data,
                    v_before_equator[1].data,
                    edges_inside[idx_after_crossing_equator:].data,
                )
            )
        )
