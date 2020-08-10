# -*- coding: utf-8 -*-
# Copyright 2018-2020 the orix developers
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

"""Vectors describing a segment of a sphere.

Each entry represents a plane normal in 3-d. Vectors can lie in, on, or outside
the spherical region.

.. image:: /_static/img/spherical-region-D3.png
   :width: 200px
   :alt: Representation of the planes comprising a spherical region.
   :align: center

Examples
--------

>>> sr = SphericalRegion([0, 0, 1])  # Region above the x-y plane
>>> v = Vector3d([(0, 0, 1), (0, 0, -1), (1, 0, 0)])
>>> v < sr
array([ True, False, False], dtype=bool)
>>> v <= sr
array([ True, False,  True], dtype=bool)

"""
import numpy as np

from orix.vector import Vector3d


class SphericalRegion(Vector3d):
    """A set of vectors representing normals segmenting a sphere.

    """

    def __gt__(self, x):
        """Returns True where x is strictly inside the region.

        Parameters
        ----------
        x : Vector3d

        Returns
        -------
        ndarray

        """
        return np.all(self.dot_outer(x) > 1e-9, axis=0)

    def __ge__(self, x):
        """Returns True if x is inside the region or one of the bounding planes.

        Parameters
        ----------
        x : Vector3d

        Returns
        -------
        ndarray

        """
        return np.all(self.dot_outer(x) > -1e-9, axis=0)
