#
# Copyright 2018-2025 the orix developers
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

import numpy as np
import pytest

import orix.crystal_map as ocm
from orix.plot import IPFColorKeyTSL
from orix.plot.direction_color_keys._util import polar_coordinates_in_sector
import orix.quaternion.symmetry as osm
import orix.vector as ove


class TestDirectionColorKeyTSL:
    def test_direction2color(self):
        ckey_oh = IPFColorKeyTSL(osm.Oh)
        ckey_oh_direction = ckey_oh.direction_color_key
        assert repr(ckey_oh_direction) == "DirectionColorKeyTSL, symmetry m-3m"

    def test_direction2color_inputs(self):
        ckey_oh = IPFColorKeyTSL(osm.Oh)
        p1 = ocm.Phase(name="fakename", point_group=osm.Oh)
        p2 = ocm.Phase(name="fakename", point_group=osm.D6h)
        arr = np.linspace(0, 5, 30).reshape(10, 3)
        v = ove.Vector3d(arr)
        m1 = ove.Miller(xyz=arr, phase=p1)
        m2 = ove.Miller(xyz=arr, phase=p2)
        rgb_key = ckey_oh.direction_color_key

        # Check Vector3ds and Millers as inputs
        rgb = rgb_key.direction2color(direction=v)
        rgb = rgb_key.direction2color(direction=m1)
        with pytest.raises(ValueError, match="'direction' has a Laue group"):
            rgb = rgb_key.direction2color(direction=m2)

    def test_triclinic(self):
        # Get RGB colors for C1. Will never reach from IPFColorKeyTSL
        # since Ci (-1) is the Laue group of C1.
        sector = osm.C1.fundamental_sector
        rgb2 = polar_coordinates_in_sector(sector, ove.Vector3d.xvector())
        assert rgb2[0].size == rgb2[1].size == 0

    @pytest.mark.parametrize(
        "symmetry, expected_shape, expected_xy_lims, expected_labels",
        [
            [osm.C1, (2000, 2000, 4), [(-1.0, 1.0), (-1.0, 1.0)], ""],
            [osm.Ci, (2000, 2000, 4), [(-1.0, 1.0), (-1.0, 1.0)], ""],
            [
                osm.C2,
                (1000, 2000, 4),
                [(-1.0, 1.0), (0.0, 1.0)],
                "[$1 0 0$][$\\bar{1} 0 0$]",
            ],
            [
                osm.S6,
                (1000, 1500, 4),
                [(-0.5, 1.0), (0, 1.0)],
                "[$2 \\bar{1} \\bar{1} 0$][$0 0 0 1$][$\\bar{1} 2 \\bar{1} 0$]",
            ],
            [
                osm.D6,
                (500, 1000, 4),
                [(0.0, 1.0), (0.0, 0.5)],
                "[$2 \\bar{1} \\bar{1} 0$][$0 0 0 1$][$1 0 \\bar{1} 0$]",
            ],
            [
                osm.Oh,
                (367, 415, 4),
                [(0.0, 0.414), (0.0, 0.366)],
                "[$1 1 1$][$1 0 1$][$0 0 1$]",
            ],
            [
                osm.Th,
                (415, 415, 4),
                [(0.0, 0.414), (0.0, 0.414)],
                "[$0 1 1$][$1 1 1$][$1 0 1$][$0 0 1$]",
            ],
        ],
    )
    @pytest.mark.slow
    def test_rgba_grid(
        self, symmetry, expected_shape, expected_xy_lims, expected_labels
    ):
        ckey = IPFColorKeyTSL(symmetry)
        ckey_dcc = ckey.direction_color_key
        rgb_grid, (xlim, ylim) = ckey_dcc._create_rgba_grid(return_extent=True)
        (x_min, x_max), (y_min, y_max) = xlim, ylim
        ax = ckey_dcc.plot(True).get_axes()[0]
        labels = "".join([x.get_text() for x in ax.texts])
        assert isinstance(rgb_grid, np.ndarray)
        assert rgb_grid.shape == expected_shape
        assert round(x_min, 3) == round(expected_xy_lims[0][0], 3)
        assert round(x_max, 3) == round(expected_xy_lims[0][1], 3)
        assert round(y_min, 3) == round(expected_xy_lims[1][0], 3)
        assert round(y_max, 3) == round(expected_xy_lims[1][1], 3)
        assert labels == expected_labels

    @pytest.mark.parametrize(
        "symmetry",
        [osm.C2, osm.D6, osm.Oh],
    )
    @pytest.mark.slow
    def test_rgba_grid_alpha(self, symmetry):
        ckey = IPFColorKeyTSL(symmetry)
        ckey_direction = ckey.direction_color_key
        rgba_grid = ckey_direction._create_rgba_grid()
        # test invalid points have alpha = 0
        assert (rgba_grid[0, 0] == (1, 1, 1, 0)).all()
        rgba_grid = ckey_direction._create_rgba_grid(alpha=0.5)
        valid = (rgba_grid != (1, 1, 1, 0)).all(axis=-1)
        assert (rgba_grid[valid][..., -1] == 0.5).all()
