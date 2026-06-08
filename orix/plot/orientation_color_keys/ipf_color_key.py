#
# Copyright 2018-2026 the orix developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix. If not, see <http://www.gnu.org/licenses/>.
#

import abc

import numpy as np

from orix.quaternion.symmetry import Symmetry
from orix.vector.miller import Miller
from orix.vector.vector3d import Vector3d

VALID_SAMPLE_DIRECTION = Vector3d | list | tuple[float, float, float]


class IPFColorKey(abc.ABC):
    def __init__(
        self, symmetry: Symmetry, direction: VALID_SAMPLE_DIRECTION | None = None
    ) -> None:
        self._symmetry = self._parse_symmetry_or_raise(symmetry)

        if direction is None:
            direction = Vector3d.zvector()
        else:
            direction = self._parse_direction_or_raise(direction)
        self._direction = direction

    # -------------------------- Properties -------------------------- #

    @property
    def direction(self) -> Vector3d:
        """Return or set the sample direction.

        Parameters
        ----------
        value : Vector3d, list, tuple[float, float, float]
            Valid sample direction.
        """
        return self._direction

    @direction.setter
    def direction(self, value: VALID_SAMPLE_DIRECTION) -> None:
        self._direction = self._parse_direction_or_raise(value)

    @property
    def symmetry(self) -> Symmetry:
        """Return or set the IPF color key symmetry."""
        return self._symmetry

    @symmetry.setter
    def symmetry(self, value: Symmetry) -> None:
        self._symmetry = self._parse_symmetry_or_raise(value)

    # ------------------------ Dunder methods ------------------------ #

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(symmetry={self.symmetry.name!r}, "
            f"direction={self.direction.data.squeeze()})"
        )

    # ------------------------ Public methods ------------------------ #

    @property
    @abc.abstractmethod
    def direction_color_key(self):  # pragma: no cover
        raise NotImplementedError

    @abc.abstractmethod
    def orientation2color(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    @abc.abstractmethod
    def plot(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    # ----------------------- Private methods ------------------------ #

    @staticmethod
    def _parse_direction_or_raise(direction: VALID_SAMPLE_DIRECTION) -> Vector3d:
        if isinstance(direction, Miller):
            raise ValueError("Sample direction cannot be a crystal direction")

        if not isinstance(direction, Vector3d):
            try:
                direction = Vector3d(np.asanyarray(direction, dtype=np.float64))
            except Exception as err:
                raise ValueError(f"Invalid sample direction {direction}") from err

        n = direction.size
        if n != 1:
            raise ValueError(f"Only one sample direction can be given, not {n}")

        return direction

    def _parse_symmetry_or_raise(self, symmetry: Symmetry) -> Symmetry:
        # Can be overwritten by subclasses
        if not isinstance(symmetry, Symmetry):
            raise ValueError(f"Invalid symmetry {symmetry}")
        return symmetry
