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

"""Symmetry element markers to plot in stereographic representations of
crystallographic point groups.

Meant to be used indirectly in
:func:`orix.plot.StereographicPlot.symmetry_marker`.
"""

import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import numpy as np

from orix.vector import Vector3d


class SymmetryMarker:
    fold = None
    _marker = None

    def __init__(self, v, size=1):
        self._vector = Vector3d(v)
        self._size = size

    @property
    def angle_deg(self):
        return np.rad2deg(self._vector.azimuth.data) + 90

    @property
    def size(self):
        return np.ones(self.n) * self._size

    @property
    def n(self):
        return self._vector.size

    def __iter__(self):
        for v, marker, size in zip(self._vector, self._marker, self.size):
            yield v, marker, size


class TwoFoldMarker(SymmetryMarker):
    fold = 2

    @property
    def size(self):
        # Assuming maximum polar angle is 90 degrees
        radial = np.tan(self._vector.polar.data / 2)
        radial = np.where(radial == 0, 1, radial)
        return self._size / np.sqrt(radial)

    @property
    def _marker(self):
        # Make an ellipse path (https://matplotlib.org/stable/api/path_api.html)
        circle = mpath.Path.circle()
        verts = np.copy(circle.vertices)  # Paths considered immutable
        verts[:, 0] *= 2
        ellipse = mpath.Path(verts, circle.codes)

        # Set up rotations of ellipse
        azimuth = self._vector.azimuth.data
        trans = [mtransforms.Affine2D().rotate(a + (np.pi / 2)) for a in azimuth]

        return [ellipse.deepcopy().transformed(i) for i in trans]


class ThreeFoldMarker(SymmetryMarker):
    fold = 3

    @property
    def _marker(self):
        return [(3, 0, angle) for angle in self.angle_deg]


class FourFoldMarker(SymmetryMarker):
    fold = 4

    @property
    def _marker(self):
        return ["D"] * self.n


class SixFoldMarker(SymmetryMarker):
    fold = 6

    @property
    def _marker(self):
        return ["h"] * self.n
