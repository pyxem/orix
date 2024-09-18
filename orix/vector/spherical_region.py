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

from orix.vector import Vector3d


class SphericalRegion(Vector3d):
    """Normals segmenting a sphere.

    Each entry represents a plane normal in 3D. Vectors can lie in, on,
    or outside the spherical region.

    .. image:: /_static/img/spherical-region-D3.png
       :width: 200px
       :alt: Representation of the planes comprising a spherical region.
       :align: center

    Examples
    --------
    >>> from orix.vector import SphericalRegion, Vector3d
    >>> sr = SphericalRegion([0, 0, 1])  # Region above the x-y plane
    >>> v = Vector3d([(0, 0, 1), (0, 0, -1), (1, 0, 0)])
    >>> v < sr
    array([ True, False, False])
    >>> v <= sr
    array([ True, False,  True])
    """

    def __gt__(self, x: Vector3d) -> np.ndarray:
        """Returns True where x is strictly inside the region.

        Parameters
        ----------
        x

        Returns
        -------
        x_out
        """
        return np.all(self.dot_outer(x) > 1e-9, axis=0)

    def __ge__(self, x: Vector3d) -> np.ndarray:
        """Returns ``True`` if ``x`` is inside the region or one of the
        bounding planes.

        Parameters
        ----------
        x

        Returns
        -------
        x_out
        """
        return np.all(self.dot_outer(x) > -1e-9, axis=0)
