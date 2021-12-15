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

import abc

from orix.vector import Vector3d


class IPFColorKey(abc.ABC):
    """Assign colors to crystal directions rotated by crystal
    orientations and projected into an inverse pole figure.

    This is an abstract class defining properties and methods required
    in derived classes.
    """

    def __init__(self, symmetry, direction=None):
        self.symmetry = symmetry
        if direction is None:
            direction = Vector3d.zvector()
        self.direction = direction

    def __repr__(self):
        return (
            f"{self.__class__.__name__}, symmetry: {self.symmetry.name}, "
            f"direction: {self.direction.data.squeeze()}"
        )

    @property
    @abc.abstractmethod
    def direction_color_key(self):
        return NotImplemented  # pragma: no cover

    @abc.abstractmethod
    def orientation2color(self, *args, **kwargs):
        return NotImplemented  # pragma: no cover

    @abc.abstractmethod
    def plot(self, *args, **kwargs):
        return NotImplemented  # pragma: no cover
