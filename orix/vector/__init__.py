#
# Copyright 2019-2025 the orix developers
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

"""Three-dimensional quantities.

Vectors can represent positions in three-dimensional space and are also
commonly associated with motion, possessing both a magnitude and a
direction. In orix they are often encountered as derived objects such as
the rotation axis of a quaternion or the normal to the bounding planes
of a spherical region.
"""

import lazy_loader

# Imports from stub file (see contributor guide for details)
__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

del lazy_loader
