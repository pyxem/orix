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

from orix.plot import IPFColorKeyTSL
from orix.plot.direction_color_keys._util import polar_coordinates_in_sector
from orix.quaternion import symmetry
from orix.vector import Vector3d


class TestDirectionColorKeyTSL:
    def test_direction2color(self):
        ckey_oh = IPFColorKeyTSL(symmetry.Oh)
        ckey_oh_direction = ckey_oh.direction_color_key
        assert repr(ckey_oh_direction) == "DirectionColorKeyTSL, symmetry m-3m"

    def test_triclinic(self):
        # Get RGB colors for C1. Will never reach from IPFColorKeyTSL
        # since Ci (-1) is the Laue group of C1.
        sector = symmetry.C1.fundamental_sector
        rgb2 = polar_coordinates_in_sector(sector, Vector3d.xvector())
        assert rgb2[0].size == rgb2[1].size == 0
