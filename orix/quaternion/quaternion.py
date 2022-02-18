# -*- coding: utf-8 -*-
# Copyright 2018-2022 the orix developers
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
import quaternion

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

    Quaternion-quaternion multiplication for two quaternions
    :math:`q_1 = (a_1, b_1, c_1, d_1)`
    and :math:`q_2 = (a_2, b_2, c_2, d_2)`
    with :math:`q_3 = (a_3, b_3, c_3, d_3) = q_1 * q_2` follows as:

    .. math::
       a_3 = (a_1 * a_2 - b_1 * b_2 - c_1 * c_2 - d_1 * d_2)

       b_3 = (a_1 * b_2 + b_1 * a_2 + c_1 * d_2 - d_1 * c_2)

       c_3 = (a_1 * c_2 - b_1 * d_2 + c_1 * a_2 + d_1 * b_2)

       d_3 = (a_1 * d_2 + b_1 * c_2 - c_1 * b_2 + d_1 * a_2)

    Quaternion-vector multiplication with a three-dimensional vector
    :math:`v = (x, y, z)` calculates a rotated vector
    :math:`v' = q * v * q^{-1}` and follows as:

    .. math::
       v'_x = x(a^2 + b^2 - c^2 - d^2) + 2z(a * c + b * d) + y(b * c - a * d)

       v'_y = y(a^2 - b^2 + c^2 - d^2) + 2x(a * d + b * c) + z(c * d - a * b)

       v'_z = z(a^2 - b^2 - c^2 + d^2) + 2y(a * b + c * d) + x(b * d - a * c)

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
        q = quaternion.from_float_array(self.data).conj()
        return Quaternion(quaternion.as_float_array(q))

    def __invert__(self):
        return self.__class__(self.conj.data / (self.norm.data**2)[..., np.newaxis])

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            q1 = quaternion.from_float_array(self.data)
            q2 = quaternion.from_float_array(other.data)
            return other.__class__(quaternion.as_float_array(q1 * q2))
        elif isinstance(other, Vector3d):
            # check broadcast shape is correct before calculation, as
            # quaternion.rotat_vectors will perform outer product
            # this keeps current __mul__ broadcast behaviour
            q1 = quaternion.from_float_array(self.data)
            v = quaternion.as_vector_part(
                (q1 * quaternion.from_vector_part(other.data)) * ~q1
            )
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

        if isinstance(other, Quaternion):
            q1 = quaternion.from_float_array(self.data)
            q2 = quaternion.from_float_array(other.data)
            # np.outer works with flattened array
            q = np.outer(q1, q2).reshape(q1.shape + q2.shape)
            return other.__class__(quaternion.as_float_array(q))
        elif isinstance(other, Vector3d):
            q = quaternion.from_float_array(self.data)
            v = quaternion.rotate_vectors(q, other.data)
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
