# -*- coding: utf-8 -*-
# Copyright 2018-2020 the orix developers
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

"""Module for generating grids in orientation spaces."""

from orix.sampling.sample_generators import get_sample_fundamental, get_sample_local
from orix.sampling.sampling_utils import uniform_SO3_sample

# Lists what will be imported when calling "from orix.sampling import *"
__all__ = [
    "get_sample_fundamental",
    "get_sample_local",
    "uniform_SO3_sample",
]
