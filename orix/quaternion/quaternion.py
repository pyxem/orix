# -*- coding: utf-8 -*-
# Copyright 2018-2023 the orix developers
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

from __future__ import annotations

from typing import Optional, Tuple, Union
import warnings

import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as SciPyRotation

from orix._util import deprecated_argument
from orix.base import Object3d
from orix.vector import AxAngle, Miller, Vector3d

# Used to round values below 1e-16 to zero
_FLOAT_EPS = np.finfo(float).eps


class Quaternion(Object3d):
    r"""Basic quaternion object.

    Quaternions support the following mathematical operations:
        - Unary negation.
        - Inversion.
        - Multiplication with other quaternions and vectors.

    Quaternion-quaternion multiplication for two quaternions
    :math:`q_1 = (a_1, b_1, c_1, d_1)`
    and :math:`q_2 = (a_2, b_2, c_2, d_2)`
    with :math:`q_3 = (a_3, b_3, c_3, d_3) = q_1 \cdot q_2` follows as:

    .. math::
       a_3 = a_1 \cdot a_2 - b_1 \cdot b_2 - c_1 \cdot c_2 - d_1 \cdot d_2

       b_3 = a_1 \cdot b_2 + b_1 \cdot a_2 + c_1 \cdot d_2 - d_1 \cdot c_2

       c_3 = a_1 \cdot c_2 - b_1 \cdot d_2 + c_1 \cdot a_2 + d_1 \cdot b_2

       d_3 = a_1 \cdot d_2 + b_1 \cdot c_2 - c_1 \cdot b_2 + d_1 \cdot a_2

    Quaternion-vector multiplication with a three-dimensional vector
    :math:`v = (x, y, z)` calculates a rotated vector
    :math:`v' = q \cdot v \cdot q^{-1}` and follows as:

    .. math::
       v'_x = x(a^2 + b^2 - c^2 - d^2) + 2(z(a \cdot c + b \cdot d) + y(b \cdot c - a \cdot d))

       v'_y = y(a^2 - b^2 + c^2 - d^2) + 2(x(a \cdot d + b \cdot c) + z(c \cdot d - a \cdot b))

       v'_z = z(a^2 - b^2 - c^2 + d^2) + 2(y(a \cdot b + c \cdot d) + x(b \cdot d - a \cdot c))
    """

    dim = 4

    @property
    def a(self) -> np.ndarray:
        """Return or set the scalar quaternion component.

        Parameters
        ----------
        value : numpy.ndarray
            Scalar quaternion component.
        """
        return self.data[..., 0]

    @a.setter
    def a(self, value: np.ndarray):
        """Set the scalar quaternion component."""
        self.data[..., 0] = value

    @property
    def b(self) -> np.ndarray:
        """Return or set the first vector quaternion component.

        Parameters
        ----------
        value : numpy.ndarray
            First vector quaternion component.
        """
        return self.data[..., 1]

    @b.setter
    def b(self, value: np.ndarray):
        """Set the first vector quaternion component."""
        self.data[..., 1] = value

    @property
    def c(self) -> np.ndarray:
        """Return or set the second vector quaternion component.

        Parameters
        ----------
        value : numpy.ndarray
            Second vector quaternion component.
        """
        return self.data[..., 2]

    @c.setter
    def c(self, value: np.ndarray):
        """Set the second vector quaternion component."""
        self.data[..., 2] = value

    @property
    def d(self) -> np.ndarray:
        """Return or set the third vector quaternion component.

        Parameters
        ----------
        value : numpy.ndarray
            Third vector quaternion component.
        """
        return self.data[..., 3]

    @d.setter
    def d(self, value: np.ndarray):
        """Set the third vector quaternion component."""
        self.data[..., 3] = value

    @property
    def antipodal(self) -> Quaternion:
        """Return the quaternions and the antipodal ones."""
        return self.__class__(np.stack([self.data, -self.data]))

    @property
    def conj(self) -> Quaternion:
        r"""Return the conjugate of this quaternion
        :math:`q^* = a - bi - cj - dk`.
        """
        q = quaternion.from_float_array(self.data).conj()
        return Quaternion(quaternion.as_float_array(q))

    def __invert__(self) -> Quaternion:
        return self.__class__(self.conj.data / (self.norm**2)[..., np.newaxis])

    def __mul__(self, other: Union[Quaternion, Vector3d]):
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

    def __neg__(self) -> Quaternion:
        return self.__class__(-self.data)

    @classmethod
    def triple_cross(cls, q1: Quaternion, q2: Quaternion, q3: Quaternion) -> Quaternion:
        """Pointwise cross product of three quaternions.

        Parameters
        ----------
        q1
            First quaternions.
        q2
            Second quaternions.
        q3
            Third quaternions.

        Returns
        -------
        q
            Quaternions resulting from the triple cross product.
        """
        q1a, q1b, q1c, q1d = q1.a, q1.b, q1.c, q1.d
        q2a, q2b, q2c, q2d = q2.a, q2.b, q2.c, q2.d
        q3a, q3b, q3c, q3d = q3.a, q3.b, q3.c, q3.d
        # fmt: off
        a = (
            + q1b * q2c * q3d
            - q1b * q3c * q2d
            - q2b * q1c * q3d
            + q2b * q3c * q1d
            + q3b * q1c * q2d
            - q3b * q2c * q1d
        )
        b = (
            + q1a * q3c * q2d
            - q1a * q2c * q3d
            + q2a * q1c * q3d
            - q2a * q3c * q1d
            - q3a * q1c * q2d
            + q3a * q2c * q1d
        )
        c = (
            + q1a * q2b * q3d
            - q1a * q3b * q2d
            - q2a * q1b * q3d
            + q2a * q3b * q1d
            + q3a * q1b * q2d
            - q3a * q2b * q1d
        )
        d = (
            + q1a * q3b * q2c
            - q1a * q2b * q3c
            + q2a * q1b * q3c
            - q2a * q3b * q1c
            - q3a * q1b * q2c
            + q3a * q2b * q1c
        )
        # fmt: on
        q = cls(np.vstack((a, b, c, d)).T)
        return q

    @classmethod
    def from_neo_euler(cls, neo_euler: "NeoEuler") -> Quaternion:
        """Create unit quaternion(s) from a neo-euler (vector)
        representation.

        Parameters
        ----------
        neo_euler
            Vector parametrization of quaternions.

        Returns
        -------
        q
            Unit quaternion(s).
        """
        s = np.sin(neo_euler.angle / 2)
        a = np.cos(neo_euler.angle / 2)
        b = s * neo_euler.axis.x
        c = s * neo_euler.axis.y
        d = s * neo_euler.axis.z
        q = cls(np.stack([a, b, c, d], axis=-1)).unit
        return q

    @classmethod
    def from_axes_angles(
        cls,
        axes: Union[np.ndarray, Vector3d, tuple, list],
        angles: Union[np.ndarray, tuple, list, float],
        degrees: bool = False,
    ) -> Quaternion:
        """Initialize from axis-angle pair(s).

        Parameters
        ----------
        axes
            Axes of rotation.
        angles
            Angles of rotation in radians (``degrees=False``) or degrees
            (``degrees=True``).
        degrees
            If ``True``, the given angles are assumed to be in degrees.
            Default is ``False``.

        Returns
        -------
        q
            Unit quaternions.

        Examples
        --------
        >>> from orix.quaternion import Quaternion
        >>> q = Quaternion.from_axes_angles((0, 0, -1), 90, degrees=True)
        >>> q
        Quaternion (1,)
        [[ 0.7071  0.      0.     -0.7071]]

        See Also
        --------
        from_neo_euler
        """
        axangle = AxAngle.from_axes_angles(axes, angles, degrees)
        q = cls.from_neo_euler(axangle).unit
        return q

    # TODO: Remove decorator, **kwargs, and use of "convention" in 0.13
    @classmethod
    @deprecated_argument("convention", "0.9", "0.13", "direction")
    def from_euler(
        cls,
        euler: Union[np.ndarray, tuple, list],
        direction: str = "lab2crystal",
        degrees: bool = False,
        **kwargs,
    ) -> Quaternion:
        """Initialize from Euler angle set(s)
        :cite:`rowenhorst2015consistent`.

        Parameters
        ----------
        euler
            Euler angles in radians (``degrees=False``) or in degrees
            (``degrees=True``) in the Bunge convention.
        direction
            Direction of the transformation, either ``"lab2crystal"``
            (default) or the inverse, ``"crystal2lab"``. The former is
            the Bunge convention. Passing ``"MTEX"`` equals the latter.
        degrees
            If ``True``, the given angles are assumed to be in degrees.
            Default is ``False``.

        Returns
        -------
        q
            Unit quaternions.
        """
        direction = direction.lower()
        if direction == "mtex" or (
            "convention" in kwargs and kwargs["convention"] == "mtex"
        ):
            # MTEX' rotations are transformations from the crystal to
            # the lab reference frames. See
            # https://mtex-toolbox.github.io/MTEXvsBungeConvention.html
            # and orix issue #215
            direction = "crystal2lab"

        directions = ["lab2crystal", "crystal2lab"]

        # processing directions
        if direction not in directions:
            raise ValueError(
                f"The chosen direction is not one of the allowed options {directions}"
            )

        if degrees:
            euler = np.deg2rad(euler)

        eu = np.asarray(euler)
        if np.any(np.abs(eu) > 4 * np.pi):
            warnings.warn("Angles are quite high, did you forget to set degrees=True?")

        n = eu.shape[:-1]
        alpha, beta, gamma = eu[..., 0], eu[..., 1], eu[..., 2]

        # Uses A.5 & A.6 from Modelling Simul. Mater. Sci. Eng. 23
        # (2015) 083501
        sigma = 0.5 * np.add(alpha, gamma)
        delta = 0.5 * np.subtract(alpha, gamma)
        c = np.cos(beta / 2)
        s = np.sin(beta / 2)

        # Using P = 1 from A.6
        q = np.zeros(n + (4,))
        q[..., 0] = c * np.cos(sigma)
        q[..., 1] = -s * np.cos(delta)
        q[..., 2] = -s * np.sin(delta)
        q[..., 3] = -c * np.sin(sigma)

        for i in [1, 2, 3, 0]:  # Flip the zero element last
            q[..., i] = np.where(q[..., 0] < 0, -q[..., i], q[..., i])

        q = cls(q).unit

        if direction == "crystal2lab":
            q = ~q

        return q

    @classmethod
    def from_matrix(cls, matrix: Union[np.ndarray, tuple, list]) -> Quaternion:
        """Create unit quaternions from the orientation matrices
        :cite:`rowenhorst2015consistent`.

        Parameters
        ----------
        matrix
            Sequence of orientation matrices with the last two
            dimensions of shape ``(3, 3)``.

        Returns
        -------
        q
            Unit quaternions.

        Examples
        --------
        >>> from orix.quaternion import Quaternion
        >>> q = Quaternion.from_matrix([np.eye(3), 2 * np.eye(3), np.diag([1, -1, -1])])
        >>> q
        Quaternion (3,)
        [[1. 0. 0. 0.]
         [1. 0. 0. 0.]
         [0. 1. 0. 0.]]
        """
        om = np.asarray(matrix)
        # Assuming (3, 3) as last two dims
        n = (1,) if om.ndim == 2 else om.shape[:-2]
        q = np.zeros(n + (4,))

        # Compute quaternion components
        q0_almost = 1 + om[..., 0, 0] + om[..., 1, 1] + om[..., 2, 2]
        q1_almost = 1 + om[..., 0, 0] - om[..., 1, 1] - om[..., 2, 2]
        q2_almost = 1 - om[..., 0, 0] + om[..., 1, 1] - om[..., 2, 2]
        q3_almost = 1 - om[..., 0, 0] - om[..., 1, 1] + om[..., 2, 2]
        q[..., 0] = 0.5 * np.sqrt(np.where(q0_almost < _FLOAT_EPS, 0, q0_almost))
        q[..., 1] = 0.5 * np.sqrt(np.where(q1_almost < _FLOAT_EPS, 0, q1_almost))
        q[..., 2] = 0.5 * np.sqrt(np.where(q2_almost < _FLOAT_EPS, 0, q2_almost))
        q[..., 3] = 0.5 * np.sqrt(np.where(q3_almost < _FLOAT_EPS, 0, q3_almost))

        # Modify component signs if necessary
        q[..., 1] = np.where(om[..., 2, 1] < om[..., 1, 2], -q[..., 1], q[..., 1])
        q[..., 2] = np.where(om[..., 0, 2] < om[..., 2, 0], -q[..., 2], q[..., 2])
        q[..., 3] = np.where(om[..., 1, 0] < om[..., 0, 1], -q[..., 3], q[..., 3])

        q = cls(q).unit

        return q

    @classmethod
    def from_scipy_rotation(cls, rotation: SciPyRotation) -> Quaternion:
        """Initialize from :class:`scipy.spatial.transform.Rotation`.

        Parameters
        ----------
        rotation
            SciPy rotation(s).

        Returns
        -------
        quaternion
            Quaternion(s).

        Notes
        -----
        The SciPy rotation is inverted to be consistent with the orix
        framework of passive rotations.

        While orix represents quaternions with the scalar as the first
        parameter, SciPy has the scalar as the last parameter.

        Examples
        --------
        >>> from orix.quaternion import Quaternion
        >>> from orix.vector import Vector3d
        >>> from scipy.spatial.transform import Rotation as SciPyRotation

        SciPy and orix represent quaternions differently

        >>> r_scipy = SciPyRotation.from_euler("ZXZ", [90, 0, 0], degrees=True)
        >>> r_scipy.as_quat()
        array([0.        , 0.        , 0.70710678, 0.70710678])
        >>> q = Quaternion.from_scipy_rotation(r_scipy)
        >>> q
        Quaternion (1,)
        [[ 0.7071  0.      0.     -0.7071]]
        >>> ~q
        Quaternion (1,)
        [[ 0.7071 -0.     -0.      0.7071]]

        SciPy and orix rotate vectors differently

        >>> v = [1, 1, 0]
        >>> r_scipy.apply(v)
        array([-1.,  1.,  0.])
        >>> q * Vector3d(v)
        Vector3d (1,)
        [[ 1. -1.  0.]]
        >>> ~q * Vector3d(v)
        Vector3d (1,)
        [[-1.  1.  0.]]
        """
        matrix = rotation.inv().as_matrix()
        return cls.from_matrix(matrix=matrix)

    @classmethod
    def from_align_vectors(
        cls,
        other: Union[Vector3d, tuple, list],
        initial: Union[Vector3d, tuple, list],
        weights: Optional[np.ndarray] = None,
        return_rmsd: bool = False,
        return_sensitivity: bool = False,
    ) -> Union[
        Quaternion,
        Tuple[Quaternion, float],
        Tuple[Quaternion, np.ndarray],
        Tuple[Quaternion, float, np.ndarray],
    ]:
        """Initialize an estimated quaternion to optimally align two
        sets of vectors.

        This method wraps
        :meth:`~scipy.spatial.transform.Rotation.align_vectors`. See
        that method for further explanations of parameters and returns.

        Parameters
        ----------
        other
            Vectors of shape ``(n,)`` in the other reference frame.
        initial
            Vectors of shape ``(n,)`` in the initial reference frame.
        weights
            Relative importance of the different vectors.
        return_rmsd
            Whether to return the (weighted) root mean square distance
            between ``other`` and ``initial`` after alignment. Default
            is ``False``.
        return_sensitivity
            Whether to return the sensitivity matrix. Default is
            ``False``.

        Returns
        -------
        estimated_quaternion
            Best estimate of the quaternion that transforms ``initial``
            to ``other``.
        rmsd
            Returned when ``return_rmsd=True``.
        sensitivity
            Returned when ``return_sensitivity=True``.

        Examples
        --------
        >>> from orix.quaternion import Quaternion
        >>> from orix.vector import Vector3d
        >>> v1 = Vector3d([[1, 0, 0], [0, 1, 0]])
        >>> v2 = Vector3d([[0, -1, 0], [0, 0, 1]])
        >>> q12 = Quaternion.from_align_vectors(v2, v1)
        >>> q12 * v1
        Vector3d (2,)
        [[ 0. -1.  0.]
         [ 0.  0.  1.]]
        >>> q21, dist = Quaternion.from_align_vectors(v1, v2, return_rmsd=True)
        >>> dist
        0.0
        >>> q21 * v2
        Vector3d (2,)
        [[1. 0. 0.]
         [0. 1. 0.]]
        """
        if not isinstance(other, Vector3d):
            other = Vector3d(other)
        if not isinstance(initial, Vector3d):
            initial = Vector3d(initial)
        vec1 = initial.unit.data
        vec2 = other.unit.data

        out = SciPyRotation.align_vectors(
            vec1, vec2, weights=weights, return_sensitivity=return_sensitivity
        )
        out = list(out)
        out[0] = cls.from_scipy_rotation(out[0])

        if not return_rmsd:
            del out[1]

        return out[0] if len(out) == 1 else tuple(out)

    @classmethod
    def random(cls, shape: Union[int, tuple] = (1,)) -> Quaternion:
        """Return random quaternions.

        Parameters
        ----------
        shape
            Shape of the quaternion instance.

        Returns
        -------
        q
            Unit quaternions.
        """
        shape = (shape,) if isinstance(shape, int) else shape
        n = int(np.prod(shape))
        q = []
        while len(q) < n:
            r = np.random.uniform(-1, 1, (3 * n, cls.dim))
            r2 = np.sum(np.square(r), axis=1)
            r = r[np.logical_and(1e-9**2 < r2, r2 <= 1)]
            q += list(r)
        q = cls(np.array(q[:n]))
        q = q.unit
        q = q.reshape(*shape)
        return q

    @classmethod
    def identity(cls, shape: Union[int, tuple] = (1,)) -> Quaternion:
        """Return identity quaternions.

        Parameters
        ----------
        shape
            Shape of the quaternion instance.

        Returns
        -------
        q
            Identity quaternions.
        """
        shape = (shape,) if isinstance(shape, int) else shape
        q = np.zeros(shape + (4,))
        q[..., 0] = 1
        return cls(q)

    def dot(self, other: Quaternion) -> np.ndarray:
        """Return the dot products of the quaternions and the other
        quaternions.

        Parameters
        ----------
        other
            Other quaternions.

        Returns
        -------
        dot_products
            Dot products.

        See Also
        --------
        Rotation.dot
        Orientation.dot

        Examples
        --------
        >>> from orix.quaternion import Quaternion
        >>> q1 = Quaternion([[1, 0, 0, 0], [0.9239, 0, 0, 0.3827]])
        >>> q2 = Quaternion([[0.9239, 0, 0, 0.3827], [0.7071, 0, 0, 0.7071]])
        >>> q1.dot(q2)
        array([0.9239    , 0.92389686])
        """
        return np.sum(self.data * other.data, axis=-1)

    def dot_outer(self, other: Quaternion) -> np.ndarray:
        """Return the dot products of all quaternions to all the other
        quaternions.

        Parameters
        ----------
        other
            Other quaternions.

        Returns
        -------
        dot_products
            Dot products.

        See Also
        --------
        Rotation.dot_outer
        Orientation.dot_outer

        Examples
        --------
        >>> from orix.quaternion import Quaternion
        >>> q1 = Quaternion([[1, 0, 0, 0], [0.9239, 0, 0, 0.3827]])
        >>> q2 = Quaternion([[0.9239, 0, 0, 0.3827], [0.7071, 0, 0, 0.7071]])
        >>> q1.dot_outer(q2)
        array([[0.9239    , 0.7071    ],
               [1.0000505 , 0.92389686]])
        """
        dots = np.tensordot(self.data, other.data, axes=(-1, -1))
        return dots

    def mean(self) -> Quaternion:
        """Return the mean quaternion with unitary weights.

        Returns
        -------
        quat_mean
            Mean quaternion.

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

    def outer(
        self,
        other: Union[Quaternion, Vector3d],
        lazy: bool = False,
        chunk_size: int = 20,
        progressbar: bool = True,
    ) -> Union[Quaternion, Vector3d]:
        """Return the outer products of the quaternions and the other
        quaternions or vectors.

        Parameters
        ----------
        other
            Another orientation or vector.
        lazy
            Whether to computer this computation using Dask. This option
            can be used to reduce memory usage when working with large
            arrays. Default is ``False``.
        chunk_size
            When using ``lazy`` computation, ``chunk_size`` represents
            the number of objects per axis for each input to include in
            each iteration of the computation. Default is 20.
        progressbar
            Whether to show a progressbar during computation if
            ``lazy=True``. Default is ``True``.

        Returns
        -------
        out
            Outer products.

        Raises
        ------
        NotImplementedError
            If ``other`` is not a quaternion, 3D vector, or a Miller
            index.
        """
        if isinstance(other, Quaternion):
            if lazy:
                darr = self._outer_dask(other, chunk_size=chunk_size)
                arr = np.empty(darr.shape)
                if progressbar:
                    with ProgressBar():
                        da.store(darr, arr)
                else:
                    da.store(darr, arr)
            else:
                q1 = quaternion.from_float_array(self.data)
                q2 = quaternion.from_float_array(other.data)
                # np.outer works with flattened array
                q = np.outer(q1, q2).reshape(q1.shape + q2.shape)
                arr = quaternion.as_float_array(q)
            return other.__class__(arr)
        elif isinstance(other, Vector3d):
            if lazy:
                darr = self._outer_dask(other, chunk_size=chunk_size)
                arr = np.empty(darr.shape)
                if progressbar:
                    with ProgressBar():
                        da.store(darr, arr)
                else:
                    da.store(darr, arr)
            else:
                q = quaternion.from_float_array(self.data)
                arr = quaternion.rotate_vectors(q, other.data)
            if isinstance(other, Miller):
                m = other.__class__(xyz=arr, phase=other.phase)
                m.coordinate_format = other.coordinate_format
                return m
            else:
                return other.__class__(arr)
        else:
            raise NotImplementedError(
                "This operation is currently not avaliable in orix, please use outer "
                "with `other` of type `Quaternion` or `Vector3d`"
            )

    # TODO: Remove decorator and **kwargs in 0.13
    @deprecated_argument("convention", since="0.9", removal="0.13")
    def to_euler(self, degrees: bool = False, **kwargs) -> np.ndarray:
        r"""Return the normalized quaternions as Euler angles in the
        Bunge convention :cite:`rowenhorst2015consistent`.

        Parameters
        ----------
        degrees
            If ``True``, the given angles are returned in degrees.
            Default is ``False``.

        Returns
        -------
        eu
            Array of Euler angles in radians (``degrees=False``) or
            degrees (``degrees=True``), in the ranges
            :math:`\phi_1 \in [0, 2\pi]`, :math:`\Phi \in [0, \pi]`, and
            :math:`\phi_1 \in [0, 2\pi]`.
        """
        # A.14 from Modelling Simul. Mater. Sci. Eng. 23 (2015) 083501
        n = self.data.shape[:-1]
        eu = np.zeros(n + (3,))

        q = self.unit
        a, b, c, d = q.a, q.b, q.c, q.d

        q03 = a**2 + d**2
        q12 = b**2 + c**2
        chi = np.sqrt(q03 * q12)

        # P = 1

        q12_is_zero = q12 == 0
        if np.sum(q12_is_zero) > 0:
            alpha = np.arctan2(-2 * a * d, a**2 - d**2)
            eu[..., 0] = np.where(q12_is_zero, alpha, eu[..., 0])
            eu[..., 1] = np.where(q12_is_zero, 0, eu[..., 1])
            eu[..., 2] = np.where(q12_is_zero, 0, eu[..., 2])

        q03_is_zero = q03 == 0
        if np.sum(q03_is_zero) > 0:
            alpha = np.arctan2(2 * b * c, b**2 - c**2)
            eu[..., 0] = np.where(q03_is_zero, alpha, eu[..., 0])
            eu[..., 1] = np.where(q03_is_zero, np.pi, eu[..., 1])
            eu[..., 2] = np.where(q03_is_zero, 0, eu[..., 2])

        if np.sum(chi != 0) > 0:
            not_zero = ~np.isclose(chi, 0)
            alpha = np.arctan2(
                np.divide(
                    b * d - a * c, chi, where=not_zero, out=np.full_like(chi, np.inf)
                ),
                np.divide(
                    -a * b - c * d, chi, where=not_zero, out=np.full_like(chi, np.inf)
                ),
            )
            beta = np.arctan2(2 * chi, q03 - q12)
            gamma = np.arctan2(
                np.divide(
                    a * c + b * d, chi, where=not_zero, out=np.full_like(chi, np.inf)
                ),
                np.divide(
                    c * d - a * b, chi, where=not_zero, out=np.full_like(chi, np.inf)
                ),
            )
            eu[..., 0] = np.where(not_zero, alpha, eu[..., 0])
            eu[..., 1] = np.where(not_zero, beta, eu[..., 1])
            eu[..., 2] = np.where(not_zero, gamma, eu[..., 2])

        # Reduce Euler angles to definition range
        eu[np.abs(eu) < _FLOAT_EPS] = 0
        eu = np.where(eu < 0, np.mod(eu + 2 * np.pi, (2 * np.pi, np.pi, 2 * np.pi)), eu)

        if degrees:
            eu = np.rad2deg(eu)

        return eu

    def to_matrix(self) -> np.ndarray:
        """Return the normalized quaternions as orientation matrices
        :cite:`rowenhorst2015consistent`.

        Returns
        -------
        om
            Array of orientation matrices after normalizing the
            quaternions.

        Examples
        --------
        >>> from orix.quaternion import Quaternion
        >>> q1 = Quaternion([[1, 0, 0, 0], [2, 0, 0, 0]])
        >>> np.allclose(q1.to_matrix(), np.eye(3))
        True
        >>> q2 = Quaternion([[0, 1, 0, 0], [0, 2, 0, 0]])
        >>> np.allclose(q2.to_matrix(), np.diag([1, -1, -1]))
        True
        """
        q = self.unit
        a, b, c, d = q.a, q.b, q.c, q.d
        om = np.zeros(self.shape + (3, 3))

        bb = b**2
        cc = c**2
        dd = d**2
        qq = a**2 - (bb + cc + dd)
        bc = b * c
        ad = a * d
        bd = b * d
        ac = a * c
        cd = c * d
        ab = a * b
        om[..., 0, 0] = qq + 2 * bb
        om[..., 0, 1] = 2 * (bc - ad)
        om[..., 0, 2] = 2 * (bd + ac)
        om[..., 1, 0] = 2 * (bc + ad)
        om[..., 1, 1] = qq + 2 * cc
        om[..., 1, 2] = 2 * (cd - ab)
        om[..., 2, 0] = 2 * (bd - ac)
        om[..., 2, 1] = 2 * (cd + ab)
        om[..., 2, 2] = qq + 2 * dd

        return om

    def _outer_dask(
        self, other: Union[Quaternion, Vector3d], chunk_size: int = 20
    ) -> da.Array:
        """Compute the product of every quaternion in this instance to
        every quaternion or vector in another instance, returned as a
        Dask array.

        For quaternion-quaternion multiplication, this is known as the
        Hamilton product.

        Parameters
        ----------
        other
            Another orientation or vector.
        chunk_size
            Number of objects per axis for each input to include in each
            iteration of the computation. Default is 20.

        Returns
        -------
        out

        Raises
        ------
        TypeError
            If ``other`` is not a quaternion or a vector.

        Notes
        -----
        For quaternion-quaternion multiplication, to create a new
        quaternion from the returned array ``out``, do
        ``q = Quaternion(out.compute())``. Likewise for
        quaternion-vector multiplication, to create a new vector from
        the returned array do ``v = Vector3d(out.compute())``.
        """
        if not isinstance(other, (Quaternion, Vector3d)):
            raise TypeError("Other must be Quaternion or Vector3d.")

        ndim1 = len(self.shape)
        ndim2 = len(other.shape)

        # Set chunk sizes
        chunks1 = (chunk_size,) * ndim1 + (-1,)
        chunks2 = (chunk_size,) * ndim2 + (-1,)

        # Dask has no dask.multiply.outer(), use dask.array.einsum
        # Summation subscripts
        str1 = "abcdefghijklm"[:ndim1]  # Max. object dimension of 13
        str2 = "nopqrstuvwxyz"[:ndim2]
        sum_over = f"...{str1},{str2}...->{str1 + str2}"

        # Get quaternion parameters as dask arrays to be computed later
        q1 = da.from_array(self.data, chunks=chunks1)
        a1, b1, c1, d1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]

        # We silence dask's einsum performance warnings for "small"
        # chunk sizes, since using the chunk sizes suggested floods
        # memory
        warnings.filterwarnings("ignore", category=da.PerformanceWarning)

        if isinstance(other, Quaternion):
            q2 = da.from_array(other.data, chunks=chunks2)
            a2, b2, c2, d2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
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
            out = da.stack([a, b, c, d], axis=-1)
        else:  # Vector3d
            v2 = da.from_array(other.data, chunks=chunks2)
            x2, y2, z2 = v2[..., 0], v2[..., 1], v2[..., 2]
            # fmt: off
            x = (
                + da.einsum(sum_over, a1**2 + b1**2 - c1**2 - d1**2, x2)
                + da.einsum(sum_over, a1 * c1 + b1 * d1, z2) * 2
                + da.einsum(sum_over, b1 * c1 - a1 * d1, y2) * 2
            )
            y = (
                + da.einsum(sum_over, a1**2 - b1**2 + c1**2 - d1**2, y2)
                + da.einsum(sum_over, a1 * d1 + b1 * c1, x2) * 2
                + da.einsum(sum_over, c1 * d1 - a1 * b1, z2) * 2
            )
            z = (
                + da.einsum(sum_over, a1**2 - b1**2 - c1**2 + d1**2, z2)
                + da.einsum(sum_over, a1 * b1 + c1 * d1, y2) * 2
                + da.einsum(sum_over, b1 * d1 - a1 * c1, x2) * 2
            )
            # fmt: on
            out = da.stack([x, y, z], axis=-1)

        new_chunks = tuple(chunks1[:-1]) + tuple(chunks2[:-1]) + (-1,)

        return out.rechunk(new_chunks)
