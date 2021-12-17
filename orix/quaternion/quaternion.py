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

import warnings

import dask.array as da
import numpy as np

from orix.base import check, Object3d
from orix.scalar import Scalar
from orix.vector import Miller, Vector3d


def check_quaternion(obj):
    return check(obj, Quaternion)


class Quaternion(Object3d):
    r"""Basic quaternion object.

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
        The conjugate of this quaternion :math:`q^* = a - bi - cj - dk`.
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
    def antipodal(self):
        return self.__class__(np.stack([self.data, -self.data], axis=0))

    @property
    def conj(self):
        a = self.a.data
        b, c, d = -self.b.data, -self.c.data, -self.d.data
        q = np.stack((a, b, c, d), axis=-1)
        return Quaternion(q)

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
            v = np.stack((x_new, y_new, z_new), axis=-1)
            if isinstance(other, Miller):
                m = other.__class__(xyz=v, phase=other.phase)
                m.coordinate_format = other.coordinate_format
                return m
            else:
                return other.__class__(v)
        return NotImplemented

    def __neg__(self):
        return self.__class__(-self.data)

    @classmethod
    def triple_cross(cls, q1, q2, q3):
        """Pointwise cross product of three quaternions.

        Parameters
        ----------
        q1, q2, q3 : orix.quaternion.Quaternion
            Three quaternions for which to find the "triple cross".

        Returns
        -------
        q : orix.quaternion.Quaternion
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

    def dot(self, other):
        """Dot product of this quaternion and the other as a
        :class:`~orix.scalar.Scalar`.
        """
        return Scalar(np.sum(self.data * other.data, axis=-1))

    def dot_outer(self, other):
        """Outer dot product of this quaternion and the other as a
        :class:`~orix.scalar.Scalar`.
        """
        dots = np.tensordot(self.data, other.data, axes=(-1, -1))
        return Scalar(dots)

    def mean(self):
        """Calculates the mean quaternion with unitary weights.

        Notes
        -----
        The method used here corresponds to Equation (13) in
        https://arc.aiaa.org/doi/pdf/10.2514/1.28949.
        """
        q = self.flatten().data.T
        qq = q.dot(q.T)
        w, v = np.linalg.eig(qq)
        w_max = np.argmax(w)
        return self.__class__(v[:, w_max])

    def outer(self, other):
        """Compute the outer product of this quaternion and the other
        quaternion or vector.

        Parameters
        ----------
        other : orix.quaternion.Quaternion or orix.vector.Vector3d

        Returns
        -------
        orix.quaternion.Quaternion or orix.vector.Vector3d
        """

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
            if isinstance(other, Miller):
                m = other.__class__(xyz=v, phase=other.phase)
                m.coordinate_format = other.coordinate_format
                return m
            else:
                return other.__class__(v)
        else:
            raise NotImplementedError(
                "This operation is currently not avaliable in orix, please use outer "
                "with `other` of type `Quaternion` or `Vector3d`"
            )

    def _outer_dask(self, other, chunk_size=20):
        """Compute the product of every quaternion in this instance to
        every quaternion in another instance, returned as a Dask array.

        This is also known as the Hamilton product.

        Parameters
        ----------
        other : orix.quaternion.Quaternion
        chunk_size : int, optional
            Number of quaternions per axis in each quaternion instance
            to include in each iteration of the computation. Default is
            20.

        Returns
        -------
        dask.array.Array

        Notes
        -----
        To get a new quaternion from the returned array `qarr`, do
        `q = Quaternion(qarr.compute())`.
        """
        ndim1 = len(self.shape)
        ndim2 = len(other.shape)

        # Set chunk sizes
        chunks1 = (chunk_size,) * ndim1 + (-1,)
        chunks2 = (chunk_size,) * ndim2 + (-1,)

        # Get quaternion parameters as dask arrays to be computed later
        q1 = da.from_array(self.data, chunks=chunks1)
        a1, b1, c1, d1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        q2 = da.from_array(other.data, chunks=chunks2)
        a2, b2, c2, d2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        # Dask has no dask.multiply.outer(), use dask.array.einsum
        # Summation subscripts
        str1 = "abcdefghijklm"[:ndim1]  # Max. object dimension of 13
        str2 = "nopqrstuvwxyz"[:ndim2]
        sum_over = f"...{str1},{str2}...->{str1 + str2}"

        # We silence dask's einsum performance warnings for "small"
        # chunk sizes, since using the chunk sizes suggested floods
        # memory
        warnings.filterwarnings("ignore", category=da.PerformanceWarning)

        # fmt: off
        a = (
            + da.einsum(sum_over, a1, a2)
            - da.einsum(sum_over, b1, b2)
            - da.einsum(sum_over, c1, c2)
            - da.einsum(sum_over, d1, d2)
        )
        b = (
            + da.einsum(sum_over, b1, a2)
            + da.einsum(sum_over, a1, b2)
            - da.einsum(sum_over, d1, c2)
            + da.einsum(sum_over, c1, d2)
        )
        c = (
            + da.einsum(sum_over, c1, a2)
            + da.einsum(sum_over, d1, b2)
            + da.einsum(sum_over, a1, c2)
            - da.einsum(sum_over, b1, d2)
        )
        d = (
            + da.einsum(sum_over, d1, a2)
            - da.einsum(sum_over, c1, b2)
            + da.einsum(sum_over, b1, c2)
            + da.einsum(sum_over, a1, d2)
        )
        # fmt: on

        new_chunks = tuple(chunks1[:-1]) + tuple(chunks2[:-1]) + (-1,)
        return da.stack((a, b, c, d), axis=-1).rechunk(new_chunks)
