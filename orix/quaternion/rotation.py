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

from __future__ import annotations

from typing import Any, Tuple, Union

import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
from scipy.special import hyp0f1

from orix.quaternion import Quaternion
from orix.vector import Vector3d


class Rotation(Quaternion):
    r"""Rotations of coordinate systems, leaving objects in place.

    Rotations :math:`R` can be parametrized in numerous ways, but in
    orix are handled as unit quaternions. Rotations can act on vectors
    or other rotations. They are often most easily visualized as being
    a turn of a certain angle :math:`\omega` about a certain axis
    :math:`\hat{\mathbf{n}}`.

    .. image:: /_static/img/rotation.png
       :width: 200px
       :alt: Rotation of an object illustrated with an axis and rotation angle.
       :align: center

    This rotation class add a sense of proper or improper rotations to
    :class:`Quaternion`. An improper rotation in orix operates on
    vectors as a rotation by the unit quaternion followed by inversion.

    See the documentation of quaternions for the applied conventions.

    Examples
    --------
    Rotate vector vA defined in coordinate system A to vector vB defined
    in coordinate system B

    >>> import numpy as np
    >>> from orix.quaternion import Quaternion
    >>> from orix.vector import Vector3d
    >>> R = Rotation.random()
    >>> vA = Vector3d.random()
    >>> vB = R * vA
    >>> np.allclose(vB.data, np.dot(R.to_matrix().squeeze(), vA.data.squeeze()))
    True

    Combine two rotations R1 and R2, applied in that order

    >>> R1, R2 = Rotation.random(2)
    >>> R12 = R2 * R1
    >>> np.allclose(
    ...     R12.to_matrix().squeeze(),
    ...     np.dot(R2.to_matrix().squeeze(), R1.to_matrix().squeeze())
    ... )
    True
    """

    def __init__(self, data: Union[np.ndarray, Rotation, list, tuple]):
        super().__init__(data)
        self._data = np.concatenate((self.data, np.zeros(self.shape + (1,))), axis=-1)
        if isinstance(data, Rotation):
            self.improper = data.improper
        with np.errstate(divide="ignore", invalid="ignore"):
            self.data /= self.norm[..., np.newaxis]

    # -------------------------- Properties -------------------------- #

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
        """Return the rotation and its antipodal."""
        R = self.__class__(np.stack([self.data, -self.data]))
        R.improper = self.improper
        return R

    # ------------------------ Dunder methods ------------------------ #

    def __mul__(
        self, other: Union[Rotation, Quaternion, Vector3d, np.ndarray, int, list]
    ):
        # Combine rotations self * other as first other, then self
        if isinstance(other, Rotation):
            Q = Quaternion(self) * Quaternion(other)
            R = other.__class__(Q)
            R.improper = np.logical_xor(self.improper, other.improper)
            return R
        if isinstance(other, Quaternion):
            Q = Quaternion(self) * other
            return Q
        if isinstance(other, Vector3d):
            v = Quaternion(self) * other
            improper = (self.improper * np.ones(other.shape)).astype(bool)
            v[improper] = -v[improper]
            return v
        if isinstance(other, int) or isinstance(other, list):  # abs(1)
            other = np.atleast_1d(other).astype(int)
            if isinstance(other, np.ndarray):
                if not np.all(abs(other) == 1):
                    raise ValueError("Rotations can only be multiplied by 1 or -1")
                R = Rotation(self.data)
                R.improper = np.logical_xor(self.improper, other == -1)
                return R
        return NotImplemented

    def __neg__(self) -> Rotation:
        R = self.__class__(self.data)
        R.improper = np.logical_not(self.improper)
        return R

    def __getitem__(self, key) -> Rotation:
        R = super().__getitem__(key)
        R.improper = self.improper[key]
        return R

    def __invert__(self) -> Rotation:
        R = super().__invert__()
        R.improper = self.improper
        return R

    def __eq__(self, other: Union[Any, Rotation]) -> bool:
        """Check if the rotations have equal shapes and values."""
        if (
            isinstance(other, Rotation)
            and self.shape == other.shape
            and np.allclose(self.data, other.data)
            and np.allclose(self.improper, other.improper)
        ):
            return True
        else:
            return False

    # ------------------------ Class methods ------------------------- #

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
        R
            Rotations.
        """
        shape = (shape,) if isinstance(shape, int) else shape
        reference = Rotation(reference)
        n = int(np.prod(shape))
        sample_size = int(alpha) * n
        R = []
        f_max = von_mises(reference, alpha, reference)
        while len(R) < n:
            R_i = cls.random(sample_size)
            f = von_mises(R_i, alpha, reference)
            x = np.random.rand(sample_size)
            R_i = R_i[x * f_max < f]
            R += list(R_i)
        return cls.stack(R[:n]).reshape(shape)

    # --------------------- Other public methods --------------------- #

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
        R
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

        R = self.flatten()

        if antipodal:
            abcd = R._differentiators()
        else:
            abcd = np.stack([R.a, R.b, R.c, R.d, R.improper], axis=-1).round(10)
        _, idx, inv = np.unique(abcd, axis=0, return_index=True, return_inverse=True)
        idx_argsort = np.argsort(idx)
        idx_sort = idx[idx_argsort]

        # Build inverse index map
        inv_map = np.empty_like(idx_argsort)
        inv_map[idx_argsort] = np.arange(idx_argsort.size)
        inv = inv_map[inv]
        dat = R[idx_sort]
        dat.improper = R.improper[idx_sort]

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
        >>> R1 = Rotation.random((5, 3))
        >>> R2 = Rotation.random((6, 2))
        >>> omega = R1.angle_with_outer(R2)
        >>> omega.shape
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
        R
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
            R = other.__class__(arr)
        else:
            R = super().outer(other)

        if isinstance(R, Rotation):
            R.improper = np.logical_xor.outer(self.improper, other.improper)
        elif isinstance(R, Vector3d):
            R[self.improper] = -R[self.improper]

        return R

    def flatten(self) -> Rotation:
        """Return a new rotation instance collapsed into one dimension.

        Returns
        -------
        R
            Rotations collapsed into one dimension.
        """
        R = super().flatten()
        R.improper = self.improper.T.flatten().T
        return R

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

    def inv(self) -> Quaternion:
        r"""Return the inverse rotations :math:`R^{-1}`."""
        return self.__invert__()

    # -------------------- Other private methods --------------------- #

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
    R
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
