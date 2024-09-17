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

"""Generation of grids on *S2* (vectors) or *SO(3)* (rotations)."""

from orix.sampling.S2_sampling import (
    sample_S2,
    sample_S2_cube_mesh,
    sample_S2_equal_area_mesh,
    sample_S2_hexagonal_mesh,
    sample_S2_icosahedral_mesh,
    sample_S2_random_mesh,
    sample_S2_uv_mesh,
)
from orix.sampling.S2_sampling import sampling_methods as sample_S2_methods
from orix.sampling.SO3_sampling import uniform_SO3_sample
from orix.sampling.sample_generators import (
    get_sample_fundamental,
    get_sample_local,
    get_sample_reduced_fundamental,
)

__all__ = [
    "get_sample_fundamental",
    "get_sample_reduced_fundamental",
    "get_sample_local",
    "sample_S2",
    "sample_S2_methods",
    "uniform_SO3_sample",
    "sample_S2_cube_mesh",
    "sample_S2_equal_area_mesh",
    "sample_S2_hexagonal_mesh",
    "sample_S2_icosahedral_mesh",
    "sample_S2_random_mesh",
    "sample_S2_uv_mesh",
]
