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

import numpy as np

from orix.base import check, Object3d
from orix.scalar import Scalar
from orix.vector import Vector3d


def check_quaternion(obj):
    return check(obj, Quaternion)


class Quaternion(Object3d):
    """Basic quaternion object.

    Quaternions support the following mathematical operations:

    - Unary negation.
    - Inversion.
    - Multiplication with other quaternions and vectors.

    Attributes
    ----------
    data : numpy.ndarray
        The numpy array containing the quaternion data.
    a, b, c, d : Scalar
        The individual elements of each vector.
    conj : Quaternion
        The conjugate of this quaternion: :math:`q^* = a - bi - cj - dk`


    """

    dim = 4

    @property
    def a(self):
        return Scalar(self.data[..., 0])

    @a.setter
    def a(self, value):
        self.data[..., 0] = value

    @property
    def b(self):
        return Scalar(self.data[..., 1])

    @b.setter
    def b(self, value):
        self.data[..., 1] = value

    @property
    def c(self):
        return Scalar(self.data[..., 2])

    @c.setter
    def c(self, value):
        self.data[..., 2] = value

    @property
    def d(self):
        return Scalar(self.data[..., 3])

    @d.setter
    def d(self, value):
        self.data[..., 3] = value

    @property
    def conj(self):
        a = self.a.data
        b, c, d = -self.b.data, -self.c.data, -self.d.data
        q = np.stack((a, b, c, d), axis=-1)
        return Quaternion(q)

    def __neg__(self):
        return self.__class__(-self.data)

    def __invert__(self):
        return self.__class__(self.conj.data / (self.norm.data ** 2)[..., np.newaxis])

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            sa, oa = self.a.data, other.a.data
            sb, ob = self.b.data, other.b.data
            sc, oc = self.c.data, other.c.data
            sd, od = self.d.data, other.d.data
            a = sa * oa - sb * ob - sc * oc - sd * od
            b = sb * oa + sa * ob - sd * oc + sc * od
            c = sc * oa + sd * ob + sa * oc - sb * od
            d = sd * oa - sc * ob + sb * oc + sa * od
            q = np.stack((a, b, c, d), axis=-1)
            return other.__class__(q)
        elif isinstance(other, Vector3d):
            a, b, c, d = self.a.data, self.b.data, self.c.data, self.d.data
            x, y, z = other.x.data, other.y.data, other.z.data
            x_new = (a ** 2 + b ** 2 - c ** 2 - d ** 2) * x + 2 * (
                (a * c + b * d) * z + (b * c - a * d) * y
            )
            y_new = (a ** 2 - b ** 2 + c ** 2 - d ** 2) * y + 2 * (
                (a * d + b * c) * x + (c * d - a * b) * z
            )
            z_new = (a ** 2 - b ** 2 - c ** 2 + d ** 2) * z + 2 * (
                (a * b + c * d) * y + (b * d - a * c) * x
            )
            return other.__class__(np.stack((x_new, y_new, z_new), axis=-1))
        return NotImplemented

    def outer(self, other):
        """Compute the outer product of this quaternion and the other object."""

        def e(x, y):
            return np.multiply.outer(x, y)

        if isinstance(other, Quaternion):
            q = np.zeros(self.shape + other.shape + (4,), dtype=float)
            sa, oa = self.data[..., 0], other.data[..., 0]
            sb, ob = self.data[..., 1], other.data[..., 1]
            sc, oc = self.data[..., 2], other.data[..., 2]
            sd, od = self.data[..., 3], other.data[..., 3]
            q[..., 0] = e(sa, oa) - e(sb, ob) - e(sc, oc) - e(sd, od)
            q[..., 1] = e(sb, oa) + e(sa, ob) - e(sd, oc) + e(sc, od)
            q[..., 2] = e(sc, oa) + e(sd, ob) + e(sa, oc) - e(sb, od)
            q[..., 3] = e(sd, oa) - e(sc, ob) + e(sb, oc) + e(sa, od)
            return other.__class__(q)
        elif isinstance(other, Vector3d):
            a, b, c, d = self.a.data, self.b.data, self.c.data, self.d.data
            x, y, z = other.x.data, other.y.data, other.z.data
            x_new = e(a ** 2 + b ** 2 - c ** 2 - d ** 2, x) + 2 * (
                e(a * c + b * d, z) + e(b * c - a * d, y)
            )
            y_new = e(a ** 2 - b ** 2 + c ** 2 - d ** 2, y) + 2 * (
                e(a * d + b * c, x) + e(c * d - a * b, z)
            )
            z_new = e(a ** 2 - b ** 2 - c ** 2 + d ** 2, z) + 2 * (
                e(a * b + c * d, y) + e(b * d - a * c, x)
            )
            v = np.stack((x_new, y_new, z_new), axis=-1)
            return other.__class__(v)
        raise NotImplementedError(
            "This operation is currently not avaliable in orix, please use outer with other of type: Quaternion or Vector3d"
        )

    def dot(self, other):
        """Scalar : the dot product of this quaternion and the other."""
        return Scalar(np.sum(self.data * other.data, axis=-1))

    def dot_outer(self, other):
        """Scalar : the outer dot product of this quaternion and the other."""
        dots = np.tensordot(self.data, other.data, axes=(-1, -1))
        return Scalar(dots)

    @classmethod
    def triple_cross(cls, q1, q2, q3):
        """Pointwise cross product of three quaternions.

        Parameters
        ----------
        q1, q2, q3 : Quaternion
            Three quaternions for which to find the "triple cross".

        Returns
        -------
        q : Quaternion

        """
        q1a, q1b, q1c, q1d = q1.a.data, q1.b.data, q1.c.data, q1.d.data
        q2a, q2b, q2c, q2d = q2.a.data, q2.b.data, q2.c.data, q2.d.data
        q3a, q3b, q3c, q3d = q3.a.data, q3.b.data, q3.c.data, q3.d.data
        a = (
            +q1b * q2c * q3d
            - q1b * q3c * q2d
            - q2b * q1c * q3d
            + q2b * q3c * q1d
            + q3b * q1c * q2d
            - q3b * q2c * q1d
        )
        b = (
            +q1a * q3c * q2d
            - q1a * q2c * q3d
            + q2a * q1c * q3d
            - q2a * q3c * q1d
            - q3a * q1c * q2d
            + q3a * q2c * q1d
        )
        c = (
            +q1a * q2b * q3d
            - q1a * q3b * q2d
            - q2a * q1b * q3d
            + q2a * q3b * q1d
            + q3a * q1b * q2d
            - q3a * q2b * q1d
        )
        d = (
            +q1a * q3b * q2c
            - q1a * q2b * q3c
            + q2a * q1b * q3c
            - q2a * q3b * q1c
            - q3a * q1b * q2c
            + q3a * q2b * q1c
        )
        q = cls(np.vstack((a, b, c, d)).T)
        return q

    @property
    def antipodal(self):
        return self.__class__(np.stack([self.data, -self.data], axis=0))

    def mean(self):
        """
        Calculates the mean quarternion with unitary weights

        Notes
        -----
        The method used here corresponds to the Equation (13) of http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf
        """
        q = self.flatten().data.T
        qq = q.dot(q.T)
        w, v = np.linalg.eig(qq)
        w_max = np.argmax(w)
        return self.__class__(v[:, w_max])
