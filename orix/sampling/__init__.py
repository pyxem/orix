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

"""Generation of grids on *S2* or *SO(3)* (rotation space)."""

from orix.sampling.sample_generators import get_sample_fundamental, get_sample_local
from orix.sampling.SO3_sampling import uniform_SO3_sample
from orix.sampling.S2_sampling import sample_S2_cube_mesh, sample_S2_uv_mesh


# Lists what will be imported when calling "from orix.sampling import *"
__all__ = [
    "get_sample_fundamental",
    "get_sample_local",
    "sample_S2_cube_mesh",
    "sample_S2_uv_mesh",
    "uniform_SO3_sample",
]
