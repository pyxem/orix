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

from orix.plot.orientation_color_keys import IPFColorKey
from orix.plot.direction_color_keys import DirectionColorKeyTSL


class IPFColorKeyTSL(IPFColorKey):
    """Assign colors to crystal directions rotated by crystal
    orientations and projected into an inverse pole figure, according to
    the Laue symmetry of the crystal.

    This is based on the TSL color key implemented in MTEX.

    Attributes
    ----------
    symmetry : orix.quaternion.Symmetry
        Laue symmetry of the crystal.
    direction : orix.vector.Vector3d
        Sample direction.
    """

    def __init__(self, symmetry, direction=None):
        super().__init__(symmetry.laue, direction=direction)

    @property
    def direction_color_key(self):
        return DirectionColorKeyTSL(self.symmetry)

    def orientation2color(self, orientation):
        # TODO: Take crystal axes into account, by using Miller instead
        #  of Vector3d
        m = orientation * self.direction
        rgb = self.direction_color_key.direction2color(m)
        return rgb

    def plot(self, return_figure=False):
        """Plot the inverse pole figure color key.

        Parameters
        ----------
        return_figure : bool, optional
            Whether to return the figure. Default is False.
        """
        return self.direction_color_key.plot(return_figure=return_figure)
