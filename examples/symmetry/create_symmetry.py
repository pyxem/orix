#
# Copyright 2018-2025 the orix developers
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

r"""
=======================
Create crystal symmetry
=======================

This example shows various ways to obtain a :class:`~orix.quaternion.symmetry.Symmetry`.
"""

########################################################################################
# The simplest way is to import pre-defined point groups.
# They are named after the Schoenflies notation and can be imported directly
from orix.quaternion.symmetry import D6h, Oh

print(Oh)  # m-3m in international/Hermann-Mauguin notation

print(D6h)  # 6/mmm

########################################################################################
# If one wanted to, one can also combine two point groups to get a third.
# Here're two versions of the orthorhombic point group *mm2* (*C2v*) with the 2-fold
# axis about different axes (importing C2v directly gives the group with the axis about
# z)
from orix.quaternion.symmetry import C2x, C2z, Csx, Csz, Symmetry

C2v_x = Symmetry.from_generators(C2x, Csz)
C2v_z = Symmetry.from_generators(C2z, Csx)

########################################################################################
# Notice the different symmetrically equivalent directions
from orix.vector import Vector3d

v = Vector3d([1.0, 1, 1])

print(C2v_x * v)
print(C2v_z * v)
