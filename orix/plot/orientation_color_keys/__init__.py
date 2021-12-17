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

"""Tools for assigning colors to crystal orientations."""

from orix.plot.orientation_color_keys.euler_color_key import EulerColorKey
from orix.plot.orientation_color_keys.ipf_color_key import IPFColorKey
from orix.plot.orientation_color_keys.ipf_color_key_tsl import IPFColorKeyTSL


__all__ = [
    "EulerColorKey",
    "IPFColorKeyTSL",
]
