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

from texpy.vector import Vector3d


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