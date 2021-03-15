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

from orix.crystal_map import Phase
from orix.vector import Miller


class TestMiller:
    pass


class TestMillerPointGroups:
    directions = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]

    def test_pg1(self):
        m = Miller(self.directions, Phase(point_group="1"))
        assert np.allclose(m.multiplicity, [1, 1, 1])
        assert np.allclose(m.symmetrise().hkl, self.directions)

    def test_pgbar1(self):
        m = Miller(self.directions, Phase(point_group="-1"))
        assert np.allclose(m.multiplicity, [1, 1, 1])
        assert np.allclose(m.symmetrise().hkl, self.directions)
