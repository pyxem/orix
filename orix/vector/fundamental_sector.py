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
        n = self.reshape(1, self.size)
        u = n.cross(n.transpose())
        return u[u <= self].unique().unit

    @property
    def center(self):
        """Center vector of the fundamental sector.

        Taken from MTEX' :code:`sphericalRegion.center`.
        """
        v = self.vertices.unique()
        n_vertices = v.size
        n_normals = self.size
        if n_normals < 2:
            return self
        elif n_vertices < 3:
            # Find the pair of maximum angle
            angles = self.angle_with(self.reshape(n_normals, 1)).data
            indices = np.argmax(angles, axis=1)
            return self[indices].mean()
        elif n_vertices < 4:
            return v.mean()
        else:
            # Avoid circular import
            from orix.sampling import uniform_SO2_sample

            v_all = uniform_SO2_sample(resolution=1)
            v = v_all[v_all < self]
            return v.mean()

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
        if self.size == 0:
            return Vector3d.empty()

        edge_steps = 200
        circles = self.get_circle(steps=edge_steps)
        edges = np.zeros((self.size * edge_steps + 3, 3))
        vertices = self.vertices

        if vertices.size == 0:
            return circles

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

        return Vector3d(edges).flatten()
