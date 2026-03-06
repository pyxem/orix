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
========================
Plot symmetry operations
========================

This example shows how stereographic projections with symmetry operation
markers can be automatically generated for the 32 crystallographic point
groups.

The ordering used here follows the one given in section 9.2 of "Structures
of Materials" (DeGraef et.al, 2nd edition, 2012). This ordering starts with
the 5 cyclic groups (C1, C2, C3, C4, and C6), followed by the 4 dihedral
groups (D2, D3, D4, and D6). Next are the same groups combined with inversion
centers (Ci, Cs, C3i, S4, and C3h), perpendicular mirror planes (C2h, C4h,
and C6h), vertical mirror planes (C2v, C3v, C4v, and C6v), and diagonal
mirror planes (D3d, D2d, and D3h). Next are groups formed from permutations
of cyclic and dihedral groups (D2h, D4h, and D6h), and finally the groups
with 3-fold rotations around the 111 axes (T, O, Th, Td, and Oh).

The plots themselves as well as their labels follow the standards given
in Table 10.2.2 of the "International Tables for Crystallography, Volume
A" (ITC). Both the nomenclature and marker styles thus differ slightly from
many textbooks, including "Structure of Materials", as there are arbitrary
convention choices in ITC regarding both Schoenflies notation and marker
style.

Orix uses Schoenflies Notation (left label above each plot) for the default
symmetry group names since they are short and always begin with a letter,
but both Schoenflies and Hermann-Mauguin (right label above each plot) names
can be used to look up symmetry groups using `PointGroups.get()`
"""

import matplotlib.pyplot as plt

import orix.plot as opl
import orix.quaternion as oqu
import orix.vector as ove

opl.register_projections()  # Register our custom Matplotlib projections

# create a list of the 32 crystallographic point groups
point_groups = oqu.symmetry.PointGroups.get_set("procedural")

# show the table of symmetry information
print(point_groups)

# prepare the plots
fig, ax = plt.subplots(
    4, 8, subplot_kw={"projection": "stereographic"}, figsize=[14, 10]
)
ax = ax.flatten()

# create a vector to mirror over axes
v = ove.Vector3d.from_polar(65, 80, degrees=True)
# Iterate through the 32 Point groups
for i, pg in enumerate(point_groups):
    pg.plot(asymmetric_vector=v, ax=ax[i])
