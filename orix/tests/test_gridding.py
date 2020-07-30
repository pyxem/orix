# -*- coding: utf-8 -*-
# Copyright 2018-2020 The pyXem developers
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

from orix.quaternion.symmetry import C1,C2,C3,C4,C6,D2,D3,D4,D6,O,T
from orix.gridding.gridding_utils import (
    create_equispaced_grid,
    get_proper_point_group,
)

def test_get_proper_point_group():
    """ Makes sure all the ints from 1 to 230 give answers"""
    for _space_group in np.arange(1, 231):
        point_group = get_proper_point_group(_space_group)
        assert point_group in [C1,C2,C3,C4,C6,D2,D3,D4,D6,O,T]
