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

import numpy as np

from orix.plot import IPFColorKeyTSL
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d


class TestIPFColorKeyTSL:
    def test_orientation2color(self):
        # Color vertices of Oh IPF red, green and close to blue
        pg_o = symmetry.O  # 432
        pg_oh = pg_o.laue  # m-3m
        ori = Orientation.from_euler(
            np.radians(((0, 0, 0), (0, 45, 0), (-45, 54.7356, 45))),
            symmetry=pg_oh,
        )
        ckey_oh = IPFColorKeyTSL(pg_o)
        assert np.allclose(ckey_oh.symmetry.data, pg_oh.data)
        assert np.allclose(ckey_oh.direction.data, (0, 0, 1))
        assert repr(ckey_oh) == "IPFColorKeyTSL, symmetry: m-3m, direction: [0 0 1]"
        fig_o = ckey_oh.plot(return_figure=True)
        ax_o = fig_o.axes[0]
        assert ax_o._symmetry.name == pg_oh.name
        rgb_oh = ckey_oh.orientation2color(ori)
        assert np.allclose(rgb_oh, ((1, 0, 0), (0, 1, 0), (0, 1 / 3, 1)), atol=0.1)

        # Color [001] and "diagonals" of 2/m IPF red, green and blue
        pg_c2 = symmetry.C2  # 2
        pg_c2h = pg_c2.laue  # 2/m
        ori2 = Orientation.from_euler(
            np.radians(((-90, -90, 0), (0, 90, -45), (0, 90, 45))),
            symmetry=pg_c2h,
        )
        ckey_c2h = IPFColorKeyTSL(pg_c2, Vector3d.xvector())
        assert np.allclose(ckey_c2h.symmetry.data, pg_c2h.data)
        assert np.allclose(ckey_c2h.direction.data, (1, 0, 0))
        assert repr(ckey_c2h) == "IPFColorKeyTSL, symmetry: 2/m, direction: [1 0 0]"
        rgb_c2h = ckey_c2h.orientation2color(ori2)
        assert np.allclose(rgb_c2h, ((1, 0, 0), (0, 1, 0), (0, 0, 1)), atol=0.1)

    def test_triclinic(self):
        # Complete circle, three vectors on equator 120 degrees apart
        pg_c1 = symmetry.C1
        ori = Orientation.from_euler(
            np.radians(((0, 90, 90), (0, 90, -30), (0, -90, 30))), symmetry=pg_c1
        )
        ckey_c1 = IPFColorKeyTSL(pg_c1)
        rgb = ckey_c1.orientation2color(ori)
        assert np.allclose(rgb, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
