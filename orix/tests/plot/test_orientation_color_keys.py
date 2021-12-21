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

import matplotlib.pyplot as plt
import numpy as np

from orix.plot import EulerColorKey, IPFColorKeyTSL
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d


class TestIPFColorKeyTSL:
    def test_orientation2color(self):
        # Color vertices of Oh IPF red, green and blue
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
        assert np.allclose(rgb_oh, ((1, 0, 0), (0, 1, 0), (0, 0, 1)), atol=0.1)

        # Color [001] and "diagonals" of 2/m IPF red, green and blue
        pg_c2 = symmetry.C2  # 2
        pg_c2h = pg_c2.laue  # 2/m
        ori2 = Orientation.from_euler(
            np.radians(((-90, -90, 0), (0, 90, -55), (0, 90, 55))),
            symmetry=pg_c2h,
        )
        ckey_c2h = IPFColorKeyTSL(pg_c2, Vector3d.xvector())
        assert np.allclose(ckey_c2h.symmetry.data, pg_c2h.data)
        assert np.allclose(ckey_c2h.direction.data, (1, 0, 0))
        assert repr(ckey_c2h) == "IPFColorKeyTSL, symmetry: 2/m, direction: [1 0 0]"
        rgb_c2h = ckey_c2h.orientation2color(ori2)
        assert np.allclose(rgb_c2h, ((1, 0, 0), (0, 1, 0.23), (0, 0.23, 1)), atol=0.2)

        # Color vertices of D3d IPF red, green and blue
        pg_d3d = symmetry.D3d  # -3m
        ori3 = Orientation.from_euler(
            np.radians(((0, 0, 0), (0, -90, 60), (0, 90, 60))),
            symmetry=pg_d3d,
        )
        ckey_d3d = IPFColorKeyTSL(pg_d3d)
        rgb_d3d = ckey_d3d.orientation2color(ori3)
        assert np.allclose(rgb_d3d, ((1, 0, 0), (0, 1, 0), (0, 0, 1)), atol=1e-2)

    def test_triclinic(self):
        # Complete circle, three vectors on equator 120 degrees apart
        pg_c1 = symmetry.C1
        ori = Orientation.from_euler(
            np.radians(((0, 90, 90), (0, 90, -30), (0, -90, 30))), symmetry=pg_c1
        )
        ckey_c1 = IPFColorKeyTSL(pg_c1)
        rgb = ckey_c1.orientation2color(ori)
        assert np.allclose(rgb, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))


class TestEulerColorKey:
    def test_orientation2color(self):
        # (2 pi, pi, 2 pi) and some random orientations
        ori = Orientation(
            (
                (0, -1, 0, 0),
                (0.4094, 0.7317, -0.4631, -0.2875),
                (-0.3885, 0.5175, -0.7589, 0.0726),
                (-0.5407, -0.7796, 0.2955, -0.1118),
                (-0.3874, 0.6708, -0.1986, 0.6004),
            )
        )

        ckey_1 = EulerColorKey(symmetry.C1)
        assert repr(ckey_1) == (
            "EulerColorKey, symmetry 1\n" "Max (phi1, Phi, phi2): (360, 180, 360)"
        )
        rgb_1 = ckey_1.orientation2color(ori)
        assert np.allclose(
            rgb_1,
            (
                (0, 1, 0),
                (0.508, 0.666, 0.687),
                (0.875, 0.741, 0.184),
                (0.410, 0.628, 0.525),
                (0.113, 0.493, 0.205),
            ),
            atol=1e-3,
        )

        ckey_432 = EulerColorKey(symmetry.O)
        assert repr(ckey_432) == (
            "EulerColorKey, symmetry 432\n" "Max (phi1, Phi, phi2): (360, 90, 90)"
        )
        rgb_432 = ckey_432.orientation2color(ori)
        assert np.allclose(
            rgb_432,
            (
                (0, 1, 0),
                (0.508, 1, 1),
                (0.875, 1, 0.737),
                (0.410, 1, 1),
                (0.113, 0.987, 0.818),
            ),
            atol=1e-3,
        )

    def test_plot(self):
        ckey_432 = EulerColorKey(symmetry.O)
        fig = ckey_432.plot(return_figure=True)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3

        labels = ["$\\phi_1$", "$\\Phi$", "$\\phi_2$"]
        angles = ckey_432.symmetry.euler_fundamental_region
        for i, ax in enumerate(fig.axes):
            if i == 0:
                assert ax.get_title() == "432"
            texts = ax.texts
            assert texts[0].get_text() == labels[i]
            assert texts[1].get_text() == str(angles[i]) + "$^{\\circ}$"

        plt.close("all")
