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
import warnings

import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
from scipy.special import hyp0f1

from orix._util import deprecated_argument
from orix.quaternion import Quaternion
from orix.vector import AxAngle, Vector3d

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
            ).round(10)
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
    def from_neo_euler(cls, neo_euler: "NeoEuler") -> Rotation:
        """Create rotations from a neo-euler (vector) representation.

        Parameters
        ----------
        neo_euler
            Vector parametrization of rotations.

        Returns
        -------
        rot
            New rotations.
        """
        s = np.sin(neo_euler.angle / 2)
        a = np.cos(neo_euler.angle / 2)
        b = s * neo_euler.axis.x
        c = s * neo_euler.axis.y
        d = s * neo_euler.axis.z
        rot = cls(np.stack([a, b, c, d], axis=-1))
        return rot

    @classmethod
    def from_axes_angles(
        cls,
        axes: Union[np.ndarray, Vector3d, tuple, list],
        angles: Union[np.ndarray, tuple, list],
    ) -> Rotation:
        """Create rotation(s) from axis-angle pair(s).

        Parameters
        ----------
        axes
            The axis of rotation.
        angles
            The angle of rotation, in radians.

        Returns
        -------
        rot
            Rotations.

        Examples
        --------
        >>> from orix.quaternion import Rotation
        >>> rot = Rotation.from_axes_angles((0, 0, -1), np.pi / 2)
        >>> rot
        Rotation (1,)
        [[ 0.7071  0.      0.     -0.7071]]

        See Also
        --------
        from_neo_euler
        """
        axangle = AxAngle.from_axes_angles(axes, angles)
        return cls.from_neo_euler(axangle)

    # TODO: Remove decorator and **kwargs in 1.0
    @deprecated_argument("convention", since="0.9", removal="1.0")
    def to_euler(self, **kwargs) -> np.ndarray:
        r"""Return the rotations as Euler angles in the Bunge convention
        :cite:`rowenhorst2015consistent`.

        Returns
        -------
        eu
            Array of Euler angles in radians, in the ranges
            :math:`\phi_1 \in [0, 2\pi]`, :math:`\Phi \in [0, \pi]`, and
            :math:`\phi_1 \in [0, 2\pi]`.
        """
        # A.14 from Modelling Simul. Mater. Sci. Eng. 23 (2015) 083501
        n = self.data.shape[:-1]
        eu = np.zeros(n + (3,))

        a, b, c, d = self.a, self.b, self.c, self.d

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

        return eu

    # TODO: Remove decorator, **kwargs, and use of "convention" in 1.0
    @classmethod
    @deprecated_argument("convention", "0.9", "1.0", "direction")
    def from_euler(
        cls,
        euler: Union[np.ndarray, list, tuple],
        direction: str = "lab2crystal",
        **kwargs,
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
        direction = direction.lower()
        if direction == "mtex" or (
            "convention" in kwargs and kwargs["convention"] == "mtex"
        ):
            # MTEX uses bunge but with lab2crystal referencing:
            # see - https://mtex-toolbox.github.io/MTEXvsBungeConvention.html
            # and orix issue #215
            direction = "crystal2lab"

        directions = ["lab2crystal", "crystal2lab"]

        # processing directions
        if direction not in directions:
            raise ValueError(
                f"The chosen direction is not one of the allowed options {directions}"
            )

        euler = np.asarray(euler)
        if np.any(np.abs(euler) > 4 * np.pi):
            warnings.warn(
                "Angles are assumed to be in radians, but degrees might have been "
                "passed."
            )
        n = euler.shape[:-1]
        alpha, beta, gamma = euler[..., 0], euler[..., 1], euler[..., 2]

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

        for i in [1, 2, 3, 0]:  # flip the zero element last
            q[..., i] = np.where(q[..., 0] < 0, -q[..., i], q[..., i])

        data = Quaternion(q)

        if direction == "crystal2lab":
            data = ~data

        rot = cls(data)
        rot.improper = np.zeros(n)
        return rot

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
        >>> rot = Rotation([1, 0, 0, 0])
        >>> np.allclose(rot.to_matrix(), np.eye(3))
        True
        >>> rot = Rotation([0, 1, 0, 0])
        >>> np.allclose(rot.to_matrix(), np.diag([1, -1, -1]))
        True
        """
        a, b, c, d = self.a, self.b, self.c, self.d
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

    @classmethod
    def from_matrix(cls, matrix: Union[np.ndarray, list, tuple]) -> Rotation:
        """Return rotations from the orientation matrices
        :cite:`rowenhorst2015consistent`.

        Parameters
        ----------
        matrix
            Array of orientation matrices.

        Examples
        --------
        >>> from orix.quaternion import Rotation
        >>> rot = Rotation.from_matrix(np.eye(3))
        >>> np.allclose(rot.data, [1, 0, 0, 0])
        True
        >>> rot = Rotation.from_matrix(np.diag([1, -1, -1]))
        >>> np.allclose(rot.data, [0, 1, 0, 0])
        True
        """
        om = np.asarray(matrix)
        # Assuming (3, 3) as last two dims
        n = (1,) if om.ndim == 2 else om.shape[:-2]
        quat = np.zeros(n + (4,))

        # Compute quaternion components
        q0_almost = 1 + om[..., 0, 0] + om[..., 1, 1] + om[..., 2, 2]
        q1_almost = 1 + om[..., 0, 0] - om[..., 1, 1] - om[..., 2, 2]
        q2_almost = 1 - om[..., 0, 0] + om[..., 1, 1] - om[..., 2, 2]
        q3_almost = 1 - om[..., 0, 0] - om[..., 1, 1] + om[..., 2, 2]
        quat[..., 0] = 0.5 * np.sqrt(np.where(q0_almost < _FLOAT_EPS, 0, q0_almost))
        quat[..., 1] = 0.5 * np.sqrt(np.where(q1_almost < _FLOAT_EPS, 0, q1_almost))
        quat[..., 2] = 0.5 * np.sqrt(np.where(q2_almost < _FLOAT_EPS, 0, q2_almost))
        quat[..., 3] = 0.5 * np.sqrt(np.where(q3_almost < _FLOAT_EPS, 0, q3_almost))

        # Modify component signs if necessary
        quat[..., 1] = np.where(
            om[..., 2, 1] < om[..., 1, 2], -quat[..., 1], quat[..., 1]
        )
        quat[..., 2] = np.where(
            om[..., 0, 2] < om[..., 2, 0], -quat[..., 2], quat[..., 2]
        )
        quat[..., 3] = np.where(
            om[..., 1, 0] < om[..., 0, 1], -quat[..., 3], quat[..., 3]
        )

        return cls(Quaternion(quat)).unit  # Normalized

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
    def random(cls, shape: Union[int, tuple] = (1,)) -> Rotation:
        """Return uniformly distributed rotations.

        Parameters
        ----------
        shape
            The shape of the required object.

        Returns
        -------
        rot
            Rotations.
        """
        shape = (shape,) if isinstance(shape, int) else shape
        n = int(np.prod(shape))
        rotations = []
        while len(rotations) < n:
            r = np.random.uniform(-1, 1, (3 * n, cls.dim))
            r2 = np.sum(np.square(r), axis=1)
            r = r[np.logical_and(1e-9**2 < r2, r2 <= 1)]
            rotations += list(r)
        return cls(np.array(rotations[:n])).reshape(*shape)

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
