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

"""Three-dimensional quantities.

Vectors can represent positions in three-dimensional space and are also
commonly associated with motion, possessing both a magnitude and a
direction. In orix they are often encountered as derived objects such as
the rotation axis of a quaternion or the normal to the bounding planes
of a spherical region.
"""

# fmt: off
# isort: off
from orix.vector.vector3d import Vector3d
from orix.vector.spherical_region import SphericalRegion
# isort: on
# fmt: on
from orix.vector.fundamental_sector import FundamentalSector
from orix.vector.miller import Miller
from orix.vector.neo_euler import AxAngle, Homochoric, NeoEuler, Rodrigues

__all__ = [
    "AxAngle",
    "FundamentalSector",
    "Homochoric",
    "Miller",
    "NeoEuler",
    "Rodrigues",
    "SphericalRegion",
    "Vector3d",
]
