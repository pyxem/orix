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

"""Point transformations of objects."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation
from scipy.special import hyp0f1

from orix.quaternion import Quaternion
from orix.vector import Vector3d

# Used to round values below 1e-16 to zero
_FLOAT_EPS = np.finfo(float).eps


class Rotation(Quaternion):
    r"""Transformations of three-dimensional space, leaving the origin in
    place.

    Rotations can be parametrized numerous ways, but in orix are handled
    as unit quaternions. Rotations can act on vectors, or other
    rotations, but not scalars. They are often most easily visualised as
    being a turn of a certain angle about a certain axis.

    .. image:: /_static/img/rotation.png
       :width: 200px
       :alt: Rotation of an object illustrated with an axis and rotation angle.
       :align: center

    Rotations can also be *improper*. An improper rotation in orix
    operates on vectors as a rotation by the unit quaternion, followed
    by inversion. Hence, a mirroring through the x-y plane can be
    considered an improper rotation of 180° about the z-axis,
    illustrated in the figure below.

    .. image:: /_static/img/inversion.png
       :width: 200px
       :alt: 180° rotation followed by inversion, leading to a mirror operation.
       :align: center

    Rotations support the following mathematical operations:
        * Unary negation.
        * Inversion.
        * Multiplication with other rotations and vectors.

    Rotations inherit all methods from :class:`Quaternion` although
    behaviour is different in some cases.

    Rotations can be converted to other parametrizations, notably the
    neo-Euler representations. See :class:`NeoEuler`.
    """

    def __init__(self, data: Union[np.ndarray, Rotation, list, tuple]):
        super().__init__(data)
        self._data = np.concatenate((self.data, np.zeros(self.shape + (1,))), axis=-1)
        if isinstance(data, Rotation):
            self.improper = data.improper
        with np.errstate(divide="ignore", invalid="ignore"):
            self.data /= self.norm[..., np.newaxis]

    @property
    def improper(self) -> np.ndarray:
        """Return ``True`` for improper rotations and ``False``
        otherwise.
        """
        return self._data[..., -1].astype(bool)

    @improper.setter
    def improper(self, value: np.ndarray):
        self._data[..., -1] = value

    @property
    def antipodal(self) -> Rotation:
        """Return this and the antipodally equivalent rotations."""
        r = self.__class__(np.stack([self.data, -self.data]))
        r.improper = self.improper
        return r

    def __mul__(
        self, other: Union[Rotation, Quaternion, Vector3d, np.ndarray, int, list]
    ):
        if isinstance(other, Rotation):
            q = Quaternion(self) * Quaternion(other)
            r = other.__class__(q)
            r.improper = np.logical_xor(self.improper, other.improper)
            return r
        if isinstance(other, Quaternion):
            q = Quaternion(self) * other
            return q
        if isinstance(other, Vector3d):
            v = Quaternion(self) * other
            improper = (self.improper * np.ones(other.shape)).astype(bool)
            v[improper] = -v[improper]
            return v
        if isinstance(other, int) or isinstance(other, list):  # has to plus/minus 1
            other = np.atleast_1d(other).astype(int)
        if isinstance(other, np.ndarray):
            assert np.all(
                abs(other) == 1
            ), "Rotations can only be multiplied by 1 or -1"
            r = Rotation(self.data)
            r.improper = np.logical_xor(self.improper, other == -1)
            return r
        return NotImplemented

    def __neg__(self) -> Rotation:
        r = self.__class__(self.data)
        r.improper = np.logical_not(self.improper)
        return r

    def __getitem__(self, key) -> Rotation:
        r = super().__getitem__(key)
        r.improper = self.improper[key]
        return r

    def __invert__(self) -> Rotation:
        r = super().__invert__()
        r.improper = self.improper
        return r

    def __eq__(self, other: Union[Any, Rotation]) -> bool:
        """Check if Rotation objects are equal by their shape and values."""
        # only return equal if shape, values, and improper arrays are equal
        if (
            isinstance(other, Rotation)
            and self.shape == other.shape
            and np.allclose(self.data, other.data)
            and np.allclose(self.improper, other.improper)
        ):
            return True
        else:
            return False

    @classmethod
    def from_axes_angles(
        cls,
        axes: Union[np.ndarray, Vector3d, tuple, list],
        angles: Union[np.ndarray, tuple, list, float],
        degrees: bool = False,
    ) -> Rotation:
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
        r
            Rotation(s).

        Examples
        --------
        >>> from orix.quaternion import Rotation
        >>> r = Rotation.from_axes_angles((0, 0, -1), 90, degrees=True)
        >>> r
        Rotation (1,)
        [[ 0.7071  0.      0.     -0.7071]]

        See Also
        --------
        from_homochoric, from_rodrigues
        """
        return super().from_axes_angles(axes, angles, degrees)

    # TODO: Remove **kwargs in 0.13
    # Deprication decorator is implemented in Quaternion
    @classmethod
    def from_euler(
        cls,
        euler: Union[np.ndarray, tuple, list],
        direction: str = "lab2crystal",
        degrees: bool = False,
        **kwargs,
    ) -> Rotation:
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
        r
            Rotation(s).
        """
        euler = np.asanyarray(euler)
        r = super().from_euler(euler, direction=direction, degrees=degrees, **kwargs)
        r.improper = np.zeros(euler.shape[:-1])
        return r

    @classmethod
    def from_matrix(cls, matrix: Union[np.ndarray, tuple, list]) -> Rotation:
        """Return rotations from the orientation matrices
        :cite:`rowenhorst2015consistent`.

        Parameters
        ----------
        matrix
            Sequence of orientation matrices with the last two
            dimensions of shape ``(3, 3)``.

        Returns
        -------
        r
            Rotations.

        Examples
        --------
        >>> from orix.quaternion import Rotation
        >>> r = Rotation.from_matrix([np.identity(3), np.diag([1, -1, -1])])
        >>> r
        Rotation (2,)
        [[1. 0. 0. 0.]
         [0. 1. 0. 0.]]
        """
        return super().from_matrix(matrix)

    @classmethod
    def from_scipy_rotation(cls, rotation: SciPyRotation) -> Rotation:
        """Return rotations(s) from
        :class:`scipy.spatial.transform.Rotation`.

        Parameters
        ----------
        rotation
            SciPy rotation(s).

        Returns
        -------
        rotation_out
            Rotation(s).

        Notes
        -----
        The SciPy rotation is inverted to be consistent with the orix
        framework of passive rotations.

        While orix represents quaternions with the scalar as the first
        parameter, SciPy has the scalar as the last parameter.

        Examples
        --------
        SciPy and orix rotate vectors differently since the SciPy
        rotation is inverted when creating an orix rotation

        >>> from orix.quaternion import Rotation
        >>> from orix.vector import Vector3d
        >>> from scipy.spatial.transform import Rotation as SciPyRotation
        >>> r_scipy = SciPyRotation.from_euler("ZXZ", [90, 0, 0], degrees=True)
        >>> r_orix = Rotation.from_scipy_rotation(r_scipy)
        >>> v = [1, 1, 0]
        >>> r_scipy.apply(v)
        array([-1.,  1.,  0.])
        >>> r_orix * Vector3d(v)
        Vector3d (1,)
        [[ 1. -1.  0.]]
        >>> ~r_orix * Vector3d(v)
        Vector3d (1,)
        [[-1.  1.  0.]]
        """
        return super().from_scipy_rotation(rotation)

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
        """Return an estimated rotation to optimally align two sets of
        vectors.

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
        estimated_rotation
            Best estimate of the rotation that transforms ``initial`` to
            ``other``.
        rmsd
            Returned when ``return_rmsd=True``.
        sensitivity
            Returned when ``return_sensitivity=True``.

        Examples
        --------
        >>> from orix.quaternion import Rotation
        >>> from orix.vector import Vector3d
        >>> v1 = Vector3d([[1, 0, 0], [0, 1, 0]])
        >>> v2 = Vector3d([[0, -1, 0], [0, 0, 1]])
        >>> r12 = Rotation.from_align_vectors(v2, v1)
        >>> r12 * v1
        Vector3d (2,)
        [[ 0. -1.  0.]
         [ 0.  0.  1.]]
        >>> r21, dist = Rotation.from_align_vectors(v1, v2, return_rmsd=True)
        >>> dist
        0.0
        >>> r21 * v2
        Vector3d (2,)
        [[1. 0. 0.]
         [0. 1. 0.]]
        """
        return super().from_align_vectors(
            other, initial, weights, return_rmsd, return_sensitivity
        )

    @classmethod
    def random(cls, shape: Union[int, tuple] = (1,)) -> Rotation:
        """Return random rotations.

        Parameters
        ----------
        shape
            Shape of the rotation instance.

        Returns
        -------
        r
            Rotations.
        """
        return super().random(shape)

    @classmethod
    def identity(cls, shape: Union[int, tuple] = (1,)) -> Rotation:
        """Return identity rotations.

        Parameters
        ----------
        shape
            Shape of the rotation instance.

        Returns
        -------
        r
            Identity rotations.
        """
        return super().identity(shape)

    @classmethod
    def random_vonmises(
        cls,
        shape: Union[int, tuple] = (1,),
        alpha: float = 1.0,
        reference: Union[list, tuple, Rotation] = (1, 0, 0, 0),
    ) -> Rotation:
        """Return random rotations with a simplified Von Mises-Fisher
        distribution.

        Parameters
        ----------
        shape
            The shape of the required object.
        alpha
            Parameter for the VM-F distribution. Lower values lead to
            "looser" distributions.
        reference
            The center of the distribution.

        Returns
        -------
        r
            Rotations.
        """
        shape = (shape,) if isinstance(shape, int) else shape
        reference = Rotation(reference)
        n = int(np.prod(shape))
        sample_size = int(alpha) * n
        r = []
        f_max = von_mises(reference, alpha, reference)
        while len(r) < n:
            r_i = cls.random(sample_size)
            f = von_mises(r_i, alpha, reference)
            x = np.random.rand(sample_size)
            r_i = r_i[x * f_max < f]
            r += list(r_i)
        return cls.stack(r[:n]).reshape(*shape)

    def unique(
        self,
        return_index: bool = False,
        return_inverse: bool = False,
        antipodal: bool = True,
    ) -> Union[
        Rotation,
        Tuple[Rotation, np.ndarray],
        Tuple[Rotation, np.ndarray, np.ndarray],
    ]:
        """Return the unique rotations from these rotations.

        Two rotations are not unique if they have the same propriety
        AND:
        - they have the same numerical value OR
        - the numerical value of one is the negative of the other

        Parameters
        ----------
        return_index
            If ``True``, will also return the indices of the (flattened)
            data where the unique entries were found.
        return_inverse
            If ``True``, will also return the indices to reconstruct the
            (flattened) data from the unique data.
        antipodal
            If ``False``, rotations representing the same transformation
            whose values are numerically different (negative) will *not*
            be considered unique.

        Returns
        -------
        r
            Unique rotations.
        idx_sort
            Indices of the flattened rotations where the unique entries
            are found. Only returned if ``return_index=True``.
        inv
            Indices to reconstruct the flattened rotations from the
            initial rotations. Only returned if ``return_inverse=True``.
        """
        if self.size == 0:
            return self.empty()

        r = self.flatten()

        if antipodal:
            abcd = r._differentiators()
        else:
            abcd = np.stack([r.a, r.b, r.c, r.d, r.improper], axis=-1).round(10)
        _, idx, inv = np.unique(abcd, axis=0, return_index=True, return_inverse=True)
        idx_argsort = np.argsort(idx)
        idx_sort = idx[idx_argsort]

        # Build inverse index map
        inv_map = np.empty_like(idx_argsort)
        inv_map[idx_argsort] = np.arange(idx_argsort.size)
        inv = inv_map[inv]
        dat = r[idx_sort]
        dat.improper = r.improper[idx_sort]

        if return_index and return_inverse:
            return dat, idx_sort, inv
        elif return_index and not return_inverse:
            return dat, idx_sort
        elif return_inverse and not return_index:
            return dat, inv
        else:
            return dat

    def angle_with(self, other: Rotation, degrees: bool = False) -> np.ndarray:
        """Return the angles of rotation transforming the rotations to
        the other rotations.

        Parameters
        ----------
        other
            Other rotations.
        degrees
            If ``True``, the angles are returned in degrees. Default is
            ``False``.

        Returns
        -------
        angles
            Angles of rotation in radians (``degrees=False``) or degrees
            (``degrees=True``).

        See Also
        --------
        angle_with
        Orientation.angle_with
        """
        other = Rotation(other)
        dot_products = self.unit.dot(other.unit)
        # Round because some dot products are slightly above 1
        dot_products = np.round(dot_products, np.finfo(dot_products.dtype).precision)
        angles = np.nan_to_num(np.arccos(2 * dot_products**2 - 1))
        if degrees:
            angles = np.rad2deg(angles)
        return angles

    def angle_with_outer(self, other: Rotation, degrees: bool = False) -> np.ndarray:
        """Return the angles of rotation transforming the rotations to
        all the other rotations.

        Parameters
        ----------
        other
            Another rotation.
        degrees
            If ``True``, the angles are returned in degrees. Default is
            ``False``.

        Returns
        -------
        angles
            Angles of rotation in radians (``degrees=False``) or degrees
            (``degrees=True``).

        Examples
        --------
        >>> from orix.quaternion import Rotation
        >>> r1 = Rotation.random((5, 3))
        >>> r2 = Rotation.random((6, 2))
        >>> dist = r1.angle_with_outer(r2)
        >>> dist.shape
        (5, 3, 6, 2)

        See Also
        --------
        angle_with
        Orientation.angle_with_outer
        """
        dot_products = self.unit.dot_outer(other.unit)
        angles = np.nan_to_num(np.arccos(2 * dot_products**2 - 1))
        if degrees:
            angles = np.rad2deg(angles)
        return angles

    def outer(
        self,
        other: Union[Rotation, Vector3d],
        lazy: bool = False,
        chunk_size: int = 20,
        progressbar: bool = True,
    ) -> Union[Rotation, Vector3d]:
        """Return the outer rotation products of the rotations and the
        other rotations or vectors.

        Parameters
        ----------
        other
            Other rotations or vectors.
        lazy
            Whether to compute the outer products using :mod:`dask`.
            This option can be used to reduce memory usage when working
            with large arrays. Default is ``False``.
        chunk_size
            Number of rotations per axis to include in each iteration of
            the computation. Default is 20. Only applies when
            ``lazy=True``. Increasing this might reduce the computation
            time at the cost of increased memory use.
        progressbar
            Whether to show a progressbar during computation if
            ``lazy=True``. Default is ``True``.

        Returns
        -------
        r
            Outer rotation products.
        """
        if lazy:
            darr = self._outer_dask(other, chunk_size=chunk_size)
            arr = np.empty(darr.shape)
            if progressbar:
                with ProgressBar():
                    da.store(darr, arr)
            else:
                da.store(darr, arr)
            r = other.__class__(arr)
        else:
            r = super().outer(other)

        if isinstance(r, Rotation):
            r.improper = np.logical_xor.outer(self.improper, other.improper)
        elif isinstance(r, Vector3d):
            r[self.improper] = -r[self.improper]

        return r

    def flatten(self) -> Rotation:
        """Return a new rotation instance collapsed into one dimension.

        Returns
        -------
        r
            Rotations collapsed into one dimension.
        """
        r = super().flatten()
        r.improper = self.improper.T.flatten().T
        return r

    def dot_outer(self, other: Rotation) -> np.ndarray:
        """Return the outer dot products of the rotations and the other
        rotations.

        Parameters
        ----------
        other
            Other rotations.

        Returns
        -------
        cosines
            Outer dot products.
        """
        dot_products = np.abs(super().dot_outer(other))
        if isinstance(other, Rotation):
            improper = self.improper.reshape(self.shape + (1,) * len(other.shape))
            i = np.logical_xor(improper, other.improper)
            dot_products = np.minimum(~i, dot_products)
        else:
            dot_products[self.improper] = 0
        return dot_products

    def to_matrix(self) -> np.ndarray:
        """Return the rotations as orientation matrices
        :cite:`rowenhorst2015consistent`.

        Returns
        -------
        om
            Array of orientation matrices.

        Examples
        --------
        >>> from orix.quaternion import Rotation
        >>> r = Rotation([[1, 0, 0, 0], [2, 0, 0, 0]])
        >>> np.allclose(r.to_matrix(), np.eye(3))
        True
        >>> r = Rotation([[0, 1, 0, 0], [0, 2, 0, 0]])
        >>> np.allclose(r.to_matrix(), np.diag([1, -1, -1]))
        True
        """
        return super().to_matrix()

    # TODO: Remove **kwargs in 0.13
    def to_euler(self, degrees: bool = False, **kwargs) -> np.ndarray:
        r"""Return the rotations as Euler angles in the Bunge convention
        :cite:`rowenhorst2015consistent`.

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
        return super().to_euler(degrees, **kwargs)

    def _differentiators(self) -> np.ndarray:
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        i = self.improper
        # fmt: off
        abcd = np.stack(
            (
                a ** 2, b ** 2, c ** 2, d ** 2,
                a * b, a * c, a * d,
                b * c, b * d,
                c * d,
                i,
            ),
            axis=-1,
        ).round(12)
        # fmt: on
        return abcd


def von_mises(
    x: Rotation, alpha: float, reference: Rotation = Rotation((1, 0, 0, 0))
) -> np.ndarray:
    r"""A vastly simplified Von Mises-Fisher distribution calculation.

    Parameters
    ----------
    x
        Rotations.
    alpha
        Lower values of alpha lead to "looser" distributions.
    reference
        Reference rotation. Default is the identity rotation.

    Returns
    -------
    r
        Rotations.

    Notes
    -----
    This simplified version of the distribution is calculated using

    .. math::
        \frac{\exp\left(2\alpha\cos\left(\omega\right)\right)}{\_0F\_1\left(\frac{N}{2}, \alpha^2\right)}

    where :math:`\omega` is the angle between orientations and :math:`N`
    is the number of relevant dimensions, in this case 3.
    """
    angle = Rotation(x).angle_with(reference)
    return np.exp(2 * alpha * np.cos(angle.data)) / hyp0f1(1.5, alpha**2)
