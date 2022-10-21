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

"""Point transformations of objects."""

from __future__ import annotations

from typing import Any, Tuple, Union

import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
from scipy.special import hyp0f1

from orix.quaternion import Quaternion
from orix.vector import Vector3d


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

    def __init__(self, data: Union[np.ndarray, Rotation, Quaternion, list, tuple]):
        super().__init__(data)
        self._data = np.concatenate((self.data, np.zeros(self.shape + (1,))), axis=-1)
        if isinstance(data, Rotation):
            self.improper = data.improper
        with np.errstate(divide="ignore", invalid="ignore"):
            self.data = self.data / self.norm[..., np.newaxis]

    def __mul__(
        self, other: Union[Rotation, Quaternion, Vector3d, np.ndarray, int, list]
    ):
        if isinstance(other, Rotation):
            quat = Quaternion(self) * Quaternion(other)
            rot = other.__class__(quat)
            i = np.logical_xor(self.improper, other.improper)
            rot.improper = i
            return rot
        if isinstance(other, Quaternion):
            quat = Quaternion(self) * other
            return quat
        if isinstance(other, Vector3d):
            vec = Quaternion(self) * other
            improper = (self.improper * np.ones(other.shape)).astype(bool)
            vec[improper] = -vec[improper]
            return vec
        if isinstance(other, int) or isinstance(other, list):  # has to plus/minus 1
            other = np.atleast_1d(other).astype(int)
        if isinstance(other, np.ndarray):
            assert np.all(
                abs(other) == 1
            ), "Rotations can only be multiplied by 1 or -1"
            rot = Rotation(self.data)
            rot.improper = np.logical_xor(self.improper, other == -1)
            return rot
        return NotImplemented

    def __neg__(self) -> Rotation:
        rot = self.__class__(self.data)
        rot.improper = np.logical_not(self.improper)
        return rot

    def __getitem__(self, key) -> Rotation:
        rot = super().__getitem__(key)
        rot.improper = self.improper[key]
        return rot

    def __invert__(self) -> Rotation:
        rot = super().__invert__()
        rot.improper = self.improper
        return rot

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
        rot
            Unique rotations.
        idx_sort
            Indices of the flattened rotations where the unique entries
            are found. Only returned if ``return_index=True``.
        inv
            Indices to reconstruct the flattened rotations from the
            initial rotations. Only returned if ``return_inverse=True``.
        """
        if len(self.data) == 0:
            return self.__class__(self.data)
        rotation = self.flatten()
        if antipodal:
            abcd = rotation._differentiators()
        else:
            abcd = np.stack(
                [
                    rotation.a,
                    rotation.b,
                    rotation.c,
                    rotation.d,
                    rotation.improper,
                ],
                axis=-1,
            ).round(12)
        _, idx, inv = np.unique(abcd, axis=0, return_index=True, return_inverse=True)
        idx_argsort = np.argsort(idx)
        idx_sort = idx[idx_argsort]
        # build inverse index map
        inv_map = np.empty_like(idx_argsort)
        inv_map[idx_argsort] = np.arange(idx_argsort.size)
        inv = inv_map[inv]
        dat = rotation[idx_sort]
        dat.improper = rotation.improper[idx_sort]
        if return_index and return_inverse:
            return dat, idx_sort, inv
        elif return_index and not return_inverse:
            return dat, idx_sort
        elif return_inverse and not return_index:
            return dat, inv
        else:
            return dat

    def _differentiators(self) -> np.ndarray:
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        i = self.improper
        abcd = np.stack(
            (
                a**2,
                b**2,
                c**2,
                d**2,
                a * b,
                a * c,
                a * d,
                b * c,
                b * d,
                c * d,
                i,
            ),
            axis=-1,
        ).round(12)
        return abcd

    def angle_with(self, other: Rotation) -> np.ndarray:
        """Return the angles of rotation transforming the rotations to
        the other rotations.

        Parameters
        ----------
        other
            Other rotations.

        Returns
        -------
        angles
            Angles of rotation.

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
        return angles

    def angle_with_outer(self, other: Rotation):
        """Return the angles of rotation transforming the rotations to
        all the other rotations.

        Parameters
        ----------
        other
            Another rotation.

        Returns
        -------
        angles
            Angles of rotation.

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
            When ``lazy=True``, ``chunk_size`` represents the number of
            rotations per axis for each input to include in each
            iteration of the computation. Default is 20.
        progressbar
            Whether to show a progressbar during computation if
            ``lazy=True``. Default is ``True``.

        Returns
        -------
        rot
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
            rot = other.__class__(arr)
        else:
            rot = super().outer(other)

        if isinstance(rot, Rotation):
            rot.improper = np.logical_xor.outer(self.improper, other.improper)
        elif isinstance(rot, Vector3d):
            rot[self.improper] = -rot[self.improper]

        return rot

    def flatten(self) -> Rotation:
        """A new object with the same data in a single column."""
        rot = super().flatten()
        rot.improper = self.improper.T.flatten().T
        return rot

    @property
    def improper(self) -> np.ndarray:
        """Return ``True`` for improper rotations and ``False``
        otherwise.
        """
        return self._data[..., -1].astype(bool)

    @improper.setter
    def improper(self, value: np.ndarray):
        self._data[..., -1] = value

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

    @classmethod
    def identity(cls, shape: tuple = (1,)) -> Rotation:
        """Create identity rotations.

        Parameters
        ----------
        shape
            The shape out of which to construct identity quaternions.

        Returns
        -------
        rot
            Identify rotations.
        """
        data = np.zeros(shape + (4,))
        data[..., 0] = 1
        return cls(data)

    # TODO: Remove **kwargs, and use of "convention" in 1.0.
    # Deprication decorator is implemented in Quaternion
    @classmethod
    def from_euler(
        cls, euler: np.ndarray, direction: str = "lab2crystal", **kwargs
    ) -> Rotation:
        """Create a rotation from an array of Euler angles in radians.

        Parameters
        ----------
        euler
            Euler angles in radians in the Bunge convention.
        direction
            "lab2crystal" (default) or "crystal2lab". "lab2crystal"
            is the Bunge convention. If "MTEX" is provided then the
            direction is "crystal2lab".
        """
        euler = np.asarray(euler)
        rot = super().from_euler(euler=euler, direction=direction, **kwargs)

        n = euler.shape[:-1]
        rot.improper = np.zeros(n)

        return rot

    @property
    def axis(self) -> Vector3d:
        """Return the axes of rotation."""
        axis = Vector3d(np.stack((self.b, self.c, self.d), axis=-1))
        a_is_zero = self.a < -1e-6
        axis[a_is_zero] = -axis[a_is_zero]
        norm_is_zero = axis.norm == 0
        axis[norm_is_zero] = Vector3d.zvector() * np.sign(self.a[norm_is_zero].data)
        axis.data = axis.data / axis.norm[..., np.newaxis]
        return axis

    @property
    def angle(self) -> np.ndarray:
        """Return the angles of rotation."""
        return 2 * np.nan_to_num(np.arccos(np.abs(self.a)))

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
        rot
            Rotations.
        """
        shape = (shape,) if isinstance(shape, int) else shape
        reference = Rotation(reference)
        n = int(np.prod(shape))
        sample_size = int(alpha) * n
        rotations = []
        f_max = von_mises(reference, alpha, reference)
        while len(rotations) < n:
            rotation = cls.random(sample_size)
            f = von_mises(rotation, alpha, reference)
            x = np.random.rand(sample_size)
            rotation = rotation[x * f_max < f]
            rotations += list(rotation)
        return cls.stack(rotations[:n]).reshape(*shape)

    @property
    def antipodal(self) -> Rotation:
        """Return this and the antipodally equivalent rotations."""
        rot = self.__class__(np.stack([self.data, -self.data]))
        rot.improper = self.improper
        return rot


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
    rot
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
