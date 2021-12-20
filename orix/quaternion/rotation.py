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

"""Point transformations of objects.

Rotations are transformations of three-dimensional space leaving the
origin in place. Rotations can be parametrized numerous ways, but in
orix are handled as unit quaternions. Rotations can act on vectors, or
other rotations, but not scalars. They are often most easily visualised
as being a turn of a certain angle about a certain axis.

.. image:: /_static/img/rotation.png
   :width: 200px
   :alt: Rotation of an object illustrated with an axis and rotation angle.
   :align: center

Rotations can also be *improper*. An improper rotation in orix operates
on vectors as a rotation by the unit quaternion, followed by inversion.
Hence, a mirroring through the x-y plane can be considered an improper
rotation of 180° about the z-axis, illustrated in the figure below.

.. image:: /_static/img/inversion.png
   :width: 200px
   :alt: 180° rotation followed by inversion, leading to a mirror operation.
   :align: center
"""

import warnings

import numpy as np
from scipy.special import hyp0f1

from orix.quaternion import Quaternion
from orix.vector import AxAngle, Vector3d
from orix.scalar import Scalar

# Used to round values below 1e-16 to zero
_FLOAT_EPS = np.finfo(float).eps


class Rotation(Quaternion):
    """Rotation object.

    Rotations support the following mathematical operations:

    - Unary negation.
    - Inversion.
    - Multiplication with other rotations and vectors.

    Rotations inherit all methods from :class:`Quaternion` although
    behaviour is different in some cases.

    Rotations can be converted to other parametrizations, notably the
    neo-Euler representations. See :class:`NeoEuler`.
    """

    def __init__(self, data):
        super().__init__(data)
        self._data = np.concatenate((self.data, np.zeros(self.shape + (1,))), axis=-1)
        if isinstance(data, Rotation):
            self.improper = data.improper
        with np.errstate(divide="ignore", invalid="ignore"):
            self.data = self.data / self.norm.data[..., np.newaxis]

    def __mul__(self, other):
        if isinstance(other, Rotation):
            q = Quaternion(self) * Quaternion(other)
            r = other.__class__(q)
            i = np.logical_xor(self.improper, other.improper)
            r.improper = i
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

    def __neg__(self):
        r = self.__class__(self.data)
        r.improper = np.logical_not(self.improper)
        return r

    def __getitem__(self, key):
        r = super().__getitem__(key)
        r.improper = self.improper[key]
        return r

    def __invert__(self):
        r = super().__invert__()
        r.improper = self.improper
        return r

    def unique(self, return_index=False, return_inverse=False, antipodal=True):
        """Returns a new object containing only this object's unique
        entries.

        Two rotations are not unique if they have the same propriety
        AND:

        - they have the same numerical value OR
        - the numerical value of one is the negative of the other

        Parameters
        ----------
        return_index : bool, optional
            If True, will also return the indices of the (flattened)
            data where the unique entries were found.
        return_inverse : bool, optional
            If True, will also return the indices to reconstruct the
            (flattened) data from the unique data.
        antipodal : bool, optional
            If False, rotations representing the same transformation
            whose values are numerically different (negative) will *not*
            be considered unique.
        """
        if len(self.data) == 0:
            return self.__class__(self.data)
        rotation = self.flatten()
        if antipodal:
            abcd = rotation._differentiators()
        else:
            abcd = np.stack(
                [
                    rotation.a.data,
                    rotation.b.data,
                    rotation.c.data,
                    rotation.d.data,
                    rotation.improper,
                ],
                axis=-1,
            ).round(6)
        _, idx, inv = np.unique(abcd, axis=0, return_index=True, return_inverse=True)
        idx_sort = np.sort(idx)
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

    def _differentiators(self):
        a = self.a.data
        b = self.b.data
        c = self.c.data
        d = self.d.data
        i = self.improper
        abcd = np.stack(
            (
                a ** 2,
                b ** 2,
                c ** 2,
                d ** 2,
                a * b,
                a * c,
                a * d,
                b * c,
                b * d,
                c * d,
                i,
            ),
            axis=-1,
        ).round(6)
        return abcd

    def angle_with(self, other):
        """The angle of rotation transforming this rotation to the
        other.

        Returns
        -------
        Scalar
        """
        other = Rotation(other)
        dp = self.unit.dot(other.unit).data
        # Round because some dot products are slightly above 1
        dp = np.round(dp, np.finfo(dp.dtype).precision)
        angles = Scalar(np.nan_to_num(np.arccos(2 * dp ** 2 - 1)))
        return angles

    def outer(self, other):
        """Compute the outer product of this rotation and the other
        rotation or vector.

        Parameters
        ----------
        other : Rotation or Vector3d

        Returns
        -------
        Rotation or Vector3d
        """
        r = super().outer(other)
        if isinstance(r, Rotation):
            r.improper = np.logical_xor.outer(self.improper, other.improper)
        if isinstance(r, Vector3d):
            r[self.improper] = -r[self.improper]
        return r

    def flatten(self):
        """A new object with the same data in a single column."""
        r = super().flatten()
        r.improper = self.improper.T.flatten().T
        return r

    @property
    def improper(self):
        """ndarray : True for improper rotations and False otherwise."""
        return self._data[..., -1].astype(bool)

    @improper.setter
    def improper(self, value):
        self._data[..., -1] = value

    def dot_outer(self, other):
        """Scalar : the outer dot product of this rotation and the other."""
        cosines = np.abs(super().dot_outer(other).data)
        if isinstance(other, Rotation):
            improper = self.improper.reshape(self.shape + (1,) * len(other.shape))
            i = np.logical_xor(improper, other.improper)
            cosines = np.minimum(~i, cosines)
        else:
            cosines[self.improper] = 0
        return Scalar(cosines)

    @classmethod
    def from_neo_euler(cls, neo_euler):
        """Creates a rotation from a neo-euler (vector) representation.

        Parameters
        ----------
        neo_euler : NeoEuler
            Vector parametrization of a rotation.
        """
        s = np.sin(neo_euler.angle.data / 2)
        a = np.cos(neo_euler.angle.data / 2)
        b = s * neo_euler.axis.x.data
        c = s * neo_euler.axis.y.data
        d = s * neo_euler.axis.z.data
        r = cls(np.stack([a, b, c, d], axis=-1))
        return r

    @classmethod
    def from_axes_angles(cls, axes, angles):
        """Creates rotation(s) from axis-angle pair(s).

        Parameters
        ----------
        axes : Vector3d or array_like
            The axis of rotation.
        angles : array_like
            The angle of rotation, in radians.

        Returns
        -------
        Rotation

        Examples
        --------
        >>> import numpy as np
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

    def to_euler(self, convention="bunge"):
        r"""Rotations as Euler angles :cite:`rowenhorst2015consistent`.

        Parameters
        ----------
        convention : str, optional
            The Euler angle convention used. Only "bunge" is supported.

        Returns
        -------
        ndarray
            Array of Euler angles in radians, in the ranges
            :math:`\phi_1 \in [0, 2\pi]`, :math:`\Phi \in [0, \pi]`, and
            :math:`\phi_1 \in [0, 2\pi]`.
        """
        # TODO: other conventions
        if convention != "bunge":
            raise ValueError(
                "The chosen convention is not supported, 'bunge' is the only option"
            )
        # A.14 from Modelling Simul. Mater. Sci. Eng. 23 (2015) 083501
        n = self.data.shape[:-1]
        e = np.zeros(n + (3,))

        a, b, c, d = self.a.data, self.b.data, self.c.data, self.d.data

        q03 = a ** 2 + d ** 2
        q12 = b ** 2 + c ** 2
        chi = np.sqrt(q03 * q12)

        # P = 1

        q12_is_zero = q12 == 0
        if np.sum(q12_is_zero) > 0:
            alpha = np.arctan2(-2 * a * d, a ** 2 - d ** 2)
            e[..., 0] = np.where(q12_is_zero, alpha, e[..., 0])
            e[..., 1] = np.where(q12_is_zero, 0, e[..., 1])
            e[..., 2] = np.where(q12_is_zero, 0, e[..., 2])

        q03_is_zero = q03 == 0
        if np.sum(q03_is_zero) > 0:
            alpha = np.arctan2(2 * b * c, b ** 2 - c ** 2)
            e[..., 0] = np.where(q03_is_zero, alpha, e[..., 0])
            e[..., 1] = np.where(q03_is_zero, np.pi, e[..., 1])
            e[..., 2] = np.where(q03_is_zero, 0, e[..., 2])

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
            e[..., 0] = np.where(not_zero, alpha, e[..., 0])
            e[..., 1] = np.where(not_zero, beta, e[..., 1])
            e[..., 2] = np.where(not_zero, gamma, e[..., 2])

        # Reduce Euler angles to definition range
        e[np.abs(e) < _FLOAT_EPS] = 0
        e = np.where(e < 0, np.mod(e + 2 * np.pi, (2 * np.pi, np.pi, 2 * np.pi)), e)

        return e

    @classmethod
    def from_euler(cls, euler, convention="bunge", direction="crystal2lab"):
        """Creates a rotation from an array of Euler angles in radians.

        Parameters
        ----------
        euler : array-like
            Euler angles in radians in the Bunge convention.
        convention : str
            "bunge" or "MTEX"
        direction : str
            "lab2crystal" or "crystal2lab", ignored if "MTEX" convention is in use
        """
        directions = ["lab2crystal", "crystal2lab"]
        conventions = ["bunge", "mtex"]

        # processing directions
        if direction not in directions:
            raise ValueError(
                f"The chosen direction is not one of the allowed options {directions}"
            )

        # processing the convention chosen

        convention = convention.lower()

        if convention not in conventions:
            raise ValueError(
                f"The chosen convention is not one of the allowed options {conventions}"
            )

        if convention == "mtex":
            # MTEX uses bunge but with lab2crystal referencing:
            # see - https://mtex-toolbox.github.io/MTEXvsBungeConvention.html
            # and orix issue #215
            direction = "lab2crystal"

        euler = np.array(euler)
        if np.any(np.abs(euler) > 9):
            warnings.warn(
                "Angles are assumed to be in radians, but degrees might have been "
                "passed"
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

        if direction == "lab2crystal":
            data = ~data

        rot = cls(data.data)
        rot.improper = np.zeros(n)
        return rot

    def to_matrix(self):
        """Rotations as orientation matrices
        :cite:`rowenhorst2015consistent`.

        Returns
        -------
        ndarray
            Array of orientation matrices.

        Examples
        --------
        >>> import numpy as np
        >>> from orix.quaternion.rotation import Rotation
        >>> r = Rotation([1, 0, 0, 0])
        >>> np.allclose(r.to_matrix(), np.eye(3))
        True
        >>> r = Rotation([0, 1, 0, 0])
        >>> np.allclose(r.to_matrix(), np.diag([1, -1, -1]))
        True
        """
        a, b, c, d = self.a.data, self.b.data, self.c.data, self.d.data
        om = np.zeros(self.shape + (3, 3))

        bb = b ** 2
        cc = c ** 2
        dd = d ** 2
        qq = a ** 2 - (bb + cc + dd)
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
    def from_matrix(cls, matrix):
        """Creates rotations from orientation matrices
        :cite:`rowenhorst2015consistent`.

        Parameters
        ----------
        matrix : array_like
            Array of orientation matrices.

        Examples
        --------
        >>> import numpy as np
        >>> from orix.quaternion import Rotation
        >>> r = Rotation.from_matrix(np.eye(3))
        >>> np.allclose(r.data, [1, 0, 0, 0])
        True
        >>> r = Rotation.from_matrix(np.diag([1, -1, -1]))
        >>> np.allclose(r.data, [0, 1, 0, 0])
        True
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

        return cls(Quaternion(q)).unit  # Normalized

    @classmethod
    def identity(cls, shape=(1,)):
        """Create identity rotations.

        Parameters
        ----------
        shape : tuple
            The shape out of which to construct identity quaternions.
        """
        data = np.zeros(shape + (4,))
        data[..., 0] = 1
        return cls(data)

    @property
    def axis(self):
        """The axis of rotation as a :class:`~orix.vector.Vector3d`."""
        axis = Vector3d(np.stack((self.b.data, self.c.data, self.d.data), axis=-1))
        a_is_zero = self.a.data < -1e-6
        axis[a_is_zero] = -axis[a_is_zero]
        norm_is_zero = axis.norm.data == 0
        axis[norm_is_zero] = Vector3d.zvector() * np.sign(self.a[norm_is_zero].data)
        axis.data = axis.data / axis.norm.data[..., np.newaxis]
        return axis

    @property
    def angle(self):
        """The angle of rotation as a :class:`~orix.scalar.Scalar`."""
        return Scalar(2 * np.nan_to_num(np.arccos(np.abs(self.a.data))))

    @classmethod
    def random(cls, shape=(1,)):
        """Uniformly distributed rotations.

        Parameters
        ----------
        shape : int or tuple of int, optional
            The shape of the required object.
        """
        shape = (shape,) if isinstance(shape, int) else shape
        n = int(np.prod(shape))
        rotations = []
        while len(rotations) < n:
            r = np.random.uniform(-1, 1, (3 * n, cls.dim))
            r2 = np.sum(np.square(r), axis=1)
            r = r[np.logical_and(1e-9 ** 2 < r2, r2 <= 1)]
            rotations += list(r)
        return cls(np.array(rotations[:n])).reshape(*shape)

    @classmethod
    def random_vonmises(cls, shape=(1,), alpha=1.0, reference=(1, 0, 0, 0)):
        """Random rotations with a simplified Von Mises-Fisher
        distribution.

        Parameters
        ----------
        shape : int or tuple of int, optional
            The shape of the required object.
        alpha : float
            Parameter for the VM-F distribution. Lower values lead to
            "looser" distributions.
        reference : Rotation
            The center of the distribution.
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
    def antipodal(self):
        """Rotation : this and antipodally equivalent rotations."""
        r = self.__class__(np.stack([self.data, -self.data], axis=0))
        r.improper = self.improper
        return r

    def _outer_dask(self, other, chunk_size=20):
        """Compute the product of every rotation in this instance to
        every rotation in another instance, returned as a Dask array.

        Parameters
        ----------
        other : orix.quaternion.Rotation
        chunk_size : int, optional
            Number of rotations per axis in each rotation instance to
            include in each iteration of the computation. Default is 20.

        Returns
        -------
        dask.array.Array

        Notes
        -----
        To get a new rotation from the returned array `rarr`, do
        `r = Rotation(rarr.compute())`.
        """
        r = super()._outer_dask(other, chunk_size=chunk_size)
        return r


def von_mises(x, alpha, reference=Rotation((1, 0, 0, 0))):
    r"""A vastly simplified Von Mises-Fisher distribution calculation.

    Parameters
    ----------
    x : Rotation
    alpha : float
        Lower values of alpha lead to "looser" distributions.
    reference : Rotation, optional

    Notes
    -----
    This simplified version of the distribution is calculated using

    .. math::
        \frac{\exp\left(2\alpha\cos\left(\omega\right)\right)}{\_0F\_1\left(\frac{N}{2}, \alpha^2\right)}

    where :math:`\omega` is the angle between orientations and :math:`N`
    is the number of relevant dimensions, in this case 3.

    Returns
    -------
    numpy.ndarray
    """
    angle = Rotation(x).angle_with(reference)
    return np.exp(2 * alpha * np.cos(angle.data)) / hyp0f1(1.5, alpha ** 2)
