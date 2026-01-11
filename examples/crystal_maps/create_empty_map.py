#
# Copyright 2018-2026 the orix developers
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

"""
========================
Create empty crystal map
========================

This example shows how to create an empty crystal map of a given shape.
By empty, we mean that it is filled with identity rotations.

This crystal map can be useful for testing.
"""

from orix.crystal_map import CrystalMap

xmap = CrystalMap.empty((5, 10))

print(xmap)
print(xmap.rotations)

xmap.plot("x")
