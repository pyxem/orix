# -*- coding: utf-8 -*-
# Copyright 2018-2021 the orix developers
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

"""The fundamental sector for a symmetry in the inverse pole figure."""

import numpy as np

from orix.vector import SphericalRegion, Vector3d


class FundamentalSector(SphericalRegion):
    """Fundamental sector for a symmetry in the inverse pole figure,
    defined by a set of (typically three) normals.
    """

    @property
    def vertices(self):
        n = self.size
        vertices = self.zero((n,))
        i_next = np.arange(1, n + 1)
        i_next[-1] = 0
        for i in range(n):
            i_n = i_next[i]
            vertices[i_n] = self[i_n].cross(self[i]).squeeze()
        return Vector3d(vertices).unit

    @property
    def center(self):
        return self.vertices.mean()

    @property
    def edges(self):
        """Unit vectors which delineates the region in the stereographic
        projection.

        They are sorted in the counter-clockwise direction around the
        sector center in the stereographic projection.

        The first edge is repeated at the end. This is done so that
        :meth:`orix.plot.StereographicPlot.plot` draws bounding lines
        without gaps.
        """
        edge_steps = 100
        circles = self.get_circle(steps=edge_steps)
        edges = np.zeros((self.size * edge_steps + 3, 3))
        vertices = self.vertices

        j = 0
        for ci, vi in zip(circles, vertices):
            edges[j] = vi.data  # Add vertex first
            j += 1
            # Only get the parts of the great circle that is within this
            # spherical region
            ci_inside = ci[ci <= self]
            ci_n = ci_inside.size
            edges[j : j + ci_n] = ci_inside.data
            j += ci_n
        edges = edges[:j]

        # Sort
        center = self.center
        vz = Vector3d.zvector()
        angle = vz.angle_with(center).data
        axis = vz.cross(center)
        edges_rotated = Vector3d(edges).rotate(axis=axis, angle=-angle)
        order = np.argsort(edges_rotated.azimuth.data)
        edges = edges[order]
        edges = np.vstack([edges, edges[0]])

        return Vector3d(edges).squeeze()
