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

"""Four-dimensional objects.

Unit quaternions are efficient objects for representing rotations, and
hence orientations.
"""

from orix.quaternion.quaternion import Quaternion  # isort: skip
from orix.quaternion.orientation import Misorientation, Orientation
from orix.quaternion.orientation_region import OrientationRegion, get_proper_groups
from orix.quaternion.rotation import Rotation, von_mises
from orix.quaternion.symmetry import Symmetry, get_distinguished_points, get_point_group

__all__ = [
    "Quaternion",
    "Rotation",
    "von_mises",
    "Misorientation",
    "Orientation",
    "get_proper_groups",
    "OrientationRegion",
    "get_distinguished_points",
    "get_point_group",
    "Symmetry",
]
