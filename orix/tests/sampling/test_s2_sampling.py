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
import pytest

from orix import sampling


class TestS2Sampling:
    @pytest.mark.parametrize("resolution, size", [(3, 7320), (4, 4140)])
    def test_uniform_S2_sample(self, resolution, size):
        v1 = sampling.uniform_S2_sample(resolution)
        assert v1.size == size
        assert np.allclose(v1.mean().data, [0, 0, 0])
        assert v1.unique().size < size
