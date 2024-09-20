# Copyright 2018-2024 the orix developers
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

from orix.quaternion.symmetry import Ci, Oh
from orix.vector import FundamentalSector, Vector3d


class TestFundamentalSector:
    # Most of the FundamentalSector class is tested in test_symmetry.py
    def test_center_from_s2_sampling(self):
        v = Vector3d([[0.5, 0, 0], [0, 0.5, 0.5], [0, 0, 1], [1, 1, 0], [0, 1, 1]])
        fs = FundamentalSector(v)

        assert fs.vertices.size == 6
        assert np.allclose(fs.center.data, [[0.534, 0.322, 0.534]], atol=1e-3)

    def test_edges(self):
        edge_steps = 1000
        fs1 = Ci.fundamental_sector
        assert np.allclose(
            fs1.edges.data, Vector3d.zvector().get_circle(steps=edge_steps).data
        )

        # Make sure that desired parts of the fundamental sector edge
        # are part of the actual edge
        fs2 = Oh.fundamental_sector
        circles = fs2.get_circle(steps=edge_steps)
        desired_edges = circles[circles <= fs2]
        actual_edges = fs2.edges
        actual_edges_data = actual_edges.data
        for edge in desired_edges.data:
            assert np.all(np.isclose(edge, actual_edges_data), axis=1).any()

        # "Joints" between the three edges are included as well
        assert desired_edges.size == 375
        assert actual_edges.size == 378
