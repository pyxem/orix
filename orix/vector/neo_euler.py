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
from typing import Union

import numpy as np

from orix.vector import Vector3d


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
    """Equal-volume mapping of the unit quaternion hemisphere.

    The homochoric vector representing a rotation with rotation angle
    :math:`\\theta` has magnitude
    :math:`\\left[\\frac{3}{4}(\\theta - \\sin\\theta)\\right]^{\\frac{1}{3}}`.

    Notes
    -----
    The homochoric transformation has no analytical inverse.
    """

    @classmethod
    def from_rotation(cls, rotation: "Rotation") -> Homochoric:
        """Create an homochoric vector from a rotation.

        Parameters
        ----------
        rotation
            Rotation.

        Returns
        -------
        vec
            Homochoric vector.
        """
        theta = rotation.angle
        n = rotation.axis
        magnitude = (0.75 * (theta - np.sin(theta))) ** (1 / 3)
        return cls(n * magnitude)

    @property
    def angle(self):
        """Calling this attribute raises an error since it cannot be
        determined analytically.
        """
        raise AttributeError(
            "The angle of a homochoric vector cannot be determined analytically."
        )


class Rodrigues(NeoEuler):
    """In Rodrigues space, straight lines map to rotations about a fixed axis.

    The Rodrigues vector representing a rotation with rotation angle
    :math:`\\theta` has magnitude :math:`\\tan\\frac{\\theta}{2}`.
    """

    @classmethod
    def from_rotation(cls, rotation: "Rotation") -> Rodrigues:
        """Create a Rodrigues vector from a rotation.

        Parameters
        ----------
        rotation
            Rotation.

        Returns
        -------
        vec
            Rodrigues vector.
        """
        a = np.float64(rotation.a)
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.stack((rotation.b / a, rotation.c / a, rotation.d / a), axis=-1)
        data[np.isnan(data)] = 0
        r = cls(data)
        return r

    @property
    def angle(self) -> np.ndarray:
        """Return the angle of the Rodrigues vector."""
        return np.arctan(self.norm) * 2


class AxAngle(NeoEuler):
    """The simplest neo-Eulerian representation.

    The Axis-Angle vector representing a rotation with rotation angle
    :math:`\\theta` has magnitude :math:`\\theta`
    """

    @classmethod
    def from_rotation(cls, rotation: "Rotation") -> AxAngle:
        """Create an axis-angle rotation from a rotation.

        Parameters
        ----------
        rotation
            Rotation.

        Returns
        -------
        vec
            Axis-angle representation of ``rotation``.
        """
        return cls((rotation.axis * rotation.angle).data)

    @property
    def angle(self):
        """Return the angle of the axis-angle rotation."""
        return self.norm

    @classmethod
    def from_axes_angles(
        cls,
        axes: Union[Vector3d, np.ndarray, list, tuple],
        angles: Union[np.ndarray, list, tuple],
    ) -> AxAngle:
        """Create new AxAngle object explicitly from the given axes and
        angles.

        Parameters
        ----------
        axes
            The axes of rotation.
        angles
            The angles of rotation, in radians.

        Returns
        -------
        vec
            Axis-angle instance of the axes and angles.
        """
        axes = Vector3d(axes).unit
        angles = np.array(angles)
        axangle_data = angles[..., np.newaxis] * axes.data
        return cls(axangle_data)
