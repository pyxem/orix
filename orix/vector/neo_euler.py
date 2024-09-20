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

"""Neo-Eulerian vectors parametrize rotations as vectors.

The rotation is specified by an axis of rotation and an angle. Different
neo-Eulerian vectors have different scaling functions applied to the angle
of rotation for different properties of the space. For example, the axis-angle
representation does not scale the angle of rotation, making it easy for direct
interpretation, whereas the Rodrigues representation applies a scaled tangent
function, such that any straight lines in Rodrigues space represent rotations
about a fixed axis.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Union

import numpy as np

from orix.vector import Vector3d

if TYPE_CHECKING:  # pragma: no cover
    from orix.quaternion import Rotation


class NeoEuler(Vector3d, abc.ABC):
    """Base class for neo-Eulerian vectors."""

    @classmethod
    @abc.abstractmethod
    def from_rotation(cls, rotation: "Rotation"):  # pragma: no cover
        """Create vectors in neo-Eulerian representation from rotations."""
        pass

    @property
    @abc.abstractmethod
    def angle(self) -> np.ndarray:  # pragma: no cover
        """Return the angles of rotation."""
        pass

    @property
    def axis(self) -> Vector3d:
        """Return the axes of rotation."""
        return Vector3d(self.unit)


class Homochoric(NeoEuler):
    r"""Equal-volume mapping of the unit quaternion hemisphere.

    The homochoric vector representing a rotation with rotation angle
    :math:`\theta` has magnitude
    :math:`\left[\frac{3}{4}(\theta - \sin\theta)\right]^{\frac{1}{3}}`.

    Notes
    -----
    The homochoric transformation has no analytical inverse.
    """

    # -------------------------- Properties -------------------------- #

    @property
    def angle(self):
        """Calling this attribute raises an error since it cannot be
        determined analytically.
        """
        raise AttributeError(
            "The angle of a homochoric vector cannot be determined analytically."
        )

    # ------------------------ Class methods ------------------------- #

    @classmethod
    def from_rotation(cls, rotation: "Rotation") -> Homochoric:
        """Create an homochoric vector from a rotation.

        Parameters
        ----------
        rotation
            Rotation.

        Returns
        -------
        v
            Homochoric vector.

        See Also
        --------
        Quaternion.to_homochoric
        """
        theta = rotation.angle
        magnitude = (0.75 * (theta - np.sin(theta))) ** (1 / 3)
        return cls(rotation.axis * magnitude)


class Rodrigues(NeoEuler):
    """In Rodrigues space, straight lines map to rotations about a fixed axis.

    The Rodrigues vector representing a rotation with rotation angle
    :math:`\\theta` has magnitude :math:`\\tan\\frac{\\theta}{2}`.
    """

    # -------------------------- Properties -------------------------- #

    @property
    def angle(self) -> np.ndarray:
        """Return the angle of the Rodrigues vector."""
        return np.arctan(self.norm) * 2

    # ------------------------ Class methods ------------------------- #

    @classmethod
    def from_rotation(cls, rotation: "Rotation") -> Rodrigues:
        """Create a Rodrigues vector from a rotation.

        Parameters
        ----------
        rotation
            Rotation.

        Returns
        -------
        v
            Rodrigues vector.

        See Also
        --------
        Quaternion.to_rodrigues
        """
        a = rotation.a.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.stack((rotation.b / a, rotation.c / a, rotation.d / a), axis=-1)
        data[np.isnan(data)] = 0
        ro = cls(data)
        return ro


class AxAngle(NeoEuler):
    r"""The simplest neo-Eulerian representation.

    The axis-angle vector representing a rotation with rotation angle
    :math:`\theta` has magnitude :math:`\theta`.
    """

    # -------------------------- Properties -------------------------- #

    @property
    def angle(self):
        """Return the angle of the axis-angle rotation."""
        return self.norm

    # ------------------------ Class methods ------------------------- #

    @classmethod
    def from_rotation(cls, rotation: "Rotation") -> AxAngle:
        """Create an axis-angle rotation from a rotation.

        Parameters
        ----------
        rotation
            Rotation.

        Returns
        -------
        v
            Axis-angle representation of ``rotation``.

        See Also
        --------
        Quaternion.to_axes_angles
        """
        return cls((rotation.axis * rotation.angle).data)

    @classmethod
    def from_axes_angles(
        cls,
        axes: Union[Vector3d, np.ndarray, list, tuple],
        angles: Union[np.ndarray, list, tuple, float],
        degrees: bool = False,
    ) -> AxAngle:
        """Initialize from axes and angles.

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
        v
            Axis-angle instance of the axes and angles.
        """
        axes = Vector3d(axes).unit
        if degrees:
            angles = np.deg2rad(angles)
        angles = np.array(angles)
        axangle_data = angles[..., np.newaxis] * axes.data
        return cls(axangle_data)
