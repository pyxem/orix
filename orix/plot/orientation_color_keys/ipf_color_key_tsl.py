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
    def __init__(self, symmetry, direction=None):
        super().__init__(symmetry.laue, direction=direction)

    @property
    def direction_color_key(self):
        return DirectionColorKeyTSL(self.symmetry)

    def orientation2color(self, orientation):
        # Doesn't take crystal axes into account! Should be Miller, not
        # Vector3d as it is now.
        m = orientation * self.direction
        rgb = self.direction_color_key.direction2color(m)
        return rgb

    def plot(self, return_figure=False):
        return self.direction_color_key.plot(return_figure=return_figure)
