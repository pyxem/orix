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

r"""
=========================
Advanced Phase operations
=========================

This example shows some additional uncommon but useful operations related to
:class:`~orix.crystal_map.Phase`.

"""

import diffpy.structure as dps

import orix.crystal_map as ocm
import orix.quaternion as oqu
import orix.vector as ove

# %%
# Directly accessing `diffpy.structure` operators
# -----------------------------------------------
#
# ORIX is primarily concerned with calculations related to point group symmetries,
# (i.e., no concern for translations). However, other projects using ORIX might
# want need this information to, for example, calculate extinction coefficients
# for a diffraction experiment. For those cases, the rotation matrices and
# translation vectors can be extracted from `Phase.space_group`.

phase = ocm.Phase(space_group=13)

for element in phase.space_group.symop_list:
    print("\nrotation:\n", element.R, "\ntranslation:", element.t)

# %%
# these can also be returned as ORIX objects.

rots = oqu.Quaternion.from_matrix([x.R for x in phase.space_group.symop_list])
vecs = ove.Vector3d([x.t for x in phase.space_group.symop_list])
