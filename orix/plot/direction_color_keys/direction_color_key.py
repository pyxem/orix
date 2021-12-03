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


class DirectionColorKey(abc.ABC):
    """Assign colors to crystal directions projected into an inverse
    pole figure.

    This is an abstract class defining properties and methods required
    in derived classes.
    """

    def __init__(self, symmetry):
        self.symmetry = symmetry

    def __repr__(self):
        return f"{self.__class__.__name__}, symmetry {self.symmetry.name}"

    @abc.abstractmethod
    def direction2color(self, direction):
        return NotImplemented  # pragma: no cover

    @abc.abstractmethod
    def plot(self):
        return NotImplemented  # pragma: no cover
