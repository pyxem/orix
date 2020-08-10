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

"""Three-dimensional quantities.

Vectors can represent positions in three-dimensional space and are also
commonly associated with motion, possessing both a magnitude and a direction.
In orix they are often encountered as derived objects such as the rotation
axis of a quaternion or the normal to the bounding planes of a spherical
region.

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    neo_euler
    spherical_region

Members
=======

"""
import numpy as np
from orix.base import check, Object3d
from orix.scalar import Scalar


def check_vector(obj):
    return check(obj, Vector3d)


class Vector3d(Object3d):
    """Vector base class.

    Vectors support the following mathematical operations:

    - Unary negation.
    - Addition to other vectors, scalars, numbers, and compatible
      array-like objects.
    - Subtraction to and from the above.
    - Multiplication to scalars, numbers, and compatible array-like objects.

    Examples
    --------
    >>> v = Vector3d((1, 2, 3))
    >>> w = Vector3d(np.array([[1, 0, 0], [0, 1, 1]]))

    >>> w.x
    Scalar (2,)
    [1 0]

    >>> v.unit
    Vector3d (1,)
    [[ 0.2673  0.5345  0.8018]]

    >>> -v
    Vector3d (1,)
    [[-1 -2 -3]]

    >>> v + w
    Vector3d (2,)
    [[2 2 3]
     [1 3 4]]

    >>> w - (2, -3)
    Vector3d (2,)
    [[-1 -2 -2]
     [ 3  4  4]]

    >>> 3 * v
    Vector3d (1,)
    [[3 6 9]]
    """

    dim = 3

    def __neg__(self):
        return self.__class__(-self.data)

    def __add__(self, other):
        if isinstance(other, Vector3d):
            return self.__class__(self.data + other.data)
        elif isinstance(other, Scalar):
            return self.__class__(self.data + other.data[..., np.newaxis])
        elif isinstance(other, (int, float)):
            return self.__class__(self.data + other)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(self.data + other[..., np.newaxis])
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Scalar):
            return self.__class__(other.data[..., np.newaxis] + self.data)
        elif isinstance(other, (int, float)):
            return self.__class__(other + self.data)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(other[..., np.newaxis] + self.data)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector3d):
            return self.__class__(self.data - other.data)
        elif isinstance(other, Scalar):
            return self.__class__(self.data - other.data[..., np.newaxis])
        elif isinstance(other, (int, float)):
            return self.__class__(self.data - other)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(self.data - other[..., np.newaxis])
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Scalar):
            return self.__class__(other.data[..., np.newaxis] - self.data)
        elif isinstance(other, (int, float)):
            return self.__class__(other - self.data)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(other[..., np.newaxis] - self.data)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Vector3d):
            raise ValueError(
                "Multiplying one vector with another is ambiguous. "
                "Try `.dot` or `.cross` instead."
            )
        elif isinstance(other, Scalar):
            return self.__class__(self.data * other.data[..., np.newaxis])
        elif isinstance(other, (int, float)):
            return self.__class__(self.data * other)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(self.data * other[..., np.newaxis])
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Scalar):
            return self.__class__(other.data[..., np.newaxis] * self.data)
        elif isinstance(other, (int, float)):
            return self.__class__(other * self.data)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(other[..., np.newaxis] * self.data)
        return NotImplemented

    def dot(self, other):
        """The dot product of a vector with another vector.

        Vectors must have compatible shape.

        Returns
        -------
        Scalar

        Examples
        --------
        >>> v = Vector3d((0, 0, 1.0))
        >>> w = Vector3d(((0, 0, 0.5), (0.4, 0.6, 0)))
        >>> v.dot(w)
        Scalar (2,)
        [ 0.5  0. ]
        >>> w.dot(v)
        Scalar (2,)
        [ 0.5  0. ]
        """
        if not isinstance(other, Vector3d):
            raise ValueError("{} is not a vector!".format(other))
        return Scalar(np.sum(self.data * other.data, axis=-1))

    def dot_outer(self, other):
        """The outer dot product of a vector with another vector.

        The dot product for every combination of vectors in `self` and `other`
        is computed.

        Returns
        -------
        Scalar

        Examples
        --------
        >>> v = Vector3d(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))  # shape = (2, )
        >>> w = Vector3d(((0.0, 0.0, 0.5), (0.4, 0.6, 0.0), (0.5, 0.5, 0.5)))  # shape = (3, )
        >>> v.dot_outer(w)
        Scalar (2, 3)
        [[ 0.5  0.   0.5]
         [ 0.   0.4  0.5]]
        >>> w.dot_outer(v)  # shape = (3, 2)
        Scalar (3, 2)
        [[ 0.5  0. ]
         [ 0.   0.4]
         [ 0.5  0.5]]

        """
        dots = np.tensordot(self.data, other.data, axes=(-1, -1))
        return Scalar(dots)

    def cross(self, other):
        """The cross product of a vector with another vector.

        Vectors must have compatible shape for broadcasting to work.

        Returns
        -------
        Vector3d
            The class of 'other' is preserved.

        Examples
        --------
        >>> v = Vector3d(((1, 0, 0), (-1, 0, 0)))
        >>> w = Vector3d((0, 1, 0))
        >>> v.cross(w)
        Vector3d (2,)
        [[ 0  0  1]
         [ 0  0 -1]]

        """
        return other.__class__(np.cross(self.data, other.data))

    @classmethod
    def from_polar(cls, theta, phi, r=1):
        """Creates a Vector3d object from polar data.
        Parameters
        ----------
        theta : array_like
            The polar angle, in radians.
        phi : array_like
            The azimuthal angle, in radians.
        r : array_like
            The radial distance. Defaults to 1 to produce unit vectors.
        Returns
        -------
        Vector3d
        """
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        z = np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        x = np.cos(phi) * np.sin(theta)
        return r * cls(np.stack((x, y, z), axis=-1))

    @classmethod
    def zero(cls, shape=(1,)):
        """Returns zero vectors in the specified shape.

        Parameters
        ----------
        shape : tuple

        Returns
        -------
        Vector3d

        """
        return cls(np.zeros(shape + (cls.dim,)))

    @classmethod
    def xvector(cls):
        """Vector3d : a single unit vector parallel to the x-direction."""
        return cls((1, 0, 0))

    @classmethod
    def yvector(cls):
        """Vector3d : a single unit vector parallel to the y-direction."""
        return cls((0, 1, 0))

    @classmethod
    def zvector(cls):
        """Vector3d : a single unit vector parallel to the z-direction."""
        return cls((0, 0, 1))

    @property
    def x(self):
        """Scalar : This vector's x data."""
        return Scalar(self.data[..., 0])

    @x.setter
    def x(self, value):
        self.data[..., 0] = value

    @property
    def y(self):
        """Scalar : This vector's y data."""
        return Scalar(self.data[..., 1])

    @y.setter
    def y(self, value):
        self.data[..., 1] = value

    @property
    def z(self):
        """Scalar : This vector's z data."""
        return Scalar(self.data[..., 2])

    @z.setter
    def z(self, value):
        self.data[..., 2] = value

    @property
    def xyz(self):
        """tuple of ndarray : This vector's components, useful for plotting."""
        return self.x.data, self.y.data, self.z.data

    def angle_with(self, other):
        """Calculate the angles between vectors in 'self' and 'other'

        Vectors must have compatible shapes for broadcasting to work.

        Returns
        -------
        Scalar
            The angle between the vectors, in radians.

        """
        cosines = np.round(self.dot(other).data / self.norm.data / other.norm.data, 9)
        return Scalar(np.arccos(cosines))

    def rotate(self, axis=None, angle=0):
        """Convenience function for rotating this vector.

        Shapes of 'axis' and 'angle' must be compatible with shape of this
        vector for broadcasting.

        Parameters
        ----------
        axis : Vector3d or array_like, optional
            The axis of rotation. Defaults to the z-vector.
        angle : array_like, optional
            The angle of rotation, in radians.

        Returns
        -------
        Vector3d
            A new vector with entries rotated.

        Examples
        --------
        >>> from math import pi
        >>> v = Vector3d((0, 1, 0))
        >>> axis = Vector3d((0, 0, 1))
        >>> angles = [0, pi/4, pi/2, 3*pi/4, pi]
        >>> v.rotate(axis=axis, angle=angles)


        """
        from orix.quaternion.rotation import Rotation
        from orix.vector.neo_euler import AxAngle

        axis = Vector3d.zvector() if axis is None else axis
        angle = 0 if angle is None else angle
        q = Rotation.from_neo_euler(AxAngle.from_axes_angles(axis, angle))
        return q * self

    @property
    def perpendicular(self):
        if np.any(self.x.data == 0) and np.any(self.y.data == 0):
            if np.any(self.z.data == 0):
                raise ValueError("Contains zero vectors!")
            return Vector3d.xvector()
        x = -self.y.data
        y = self.x.data
        z = np.zeros_like(x)
        return Vector3d(np.stack((x, y, z), axis=-1))

    def get_nearest(self, x, inclusive=False, tiebreak=None):
        """The vector among x with the smallest angle to this one.

        Parameters
        ----------
        x : Vector3d
        inclusive : bool
            if False (default) vectors exactly parallel to this will not be
            considered.
        tiebreak : Vector3d
            If multiple vectors are equally close to this one, `tiebreak` will
            be used as a secondary comparison. By default equal to (0, 0, 1).

        Returns
        -------
        Vector3d

        """
        assert self.size == 1, "`get_nearest` only works for single vectors."
        tiebreak = Vector3d.zvector() if tiebreak is None else tiebreak
        eps = 1e-9 if inclusive else 0.0
        cosines = x.dot(self).data
        mask = np.logical_and(-1 - eps < cosines, cosines < 1 + eps)
        x = x[mask]
        if x.size == 0:
            return Vector3d.empty()
        cosines = cosines[mask]
        verticality = x.dot(tiebreak).data
        order = np.lexsort((cosines, verticality))
        return x[order[-1]]

    @property
    def _tuples(self):
        """set of tuple : the set of comparable vectors."""
        s = self.flatten()
        tuples = set([tuple(d) for d in s.data])
        return tuples

    def mean(self):
        axis = tuple(range(self.data_dim))
        return self.__class__(self.data.mean(axis=axis))
