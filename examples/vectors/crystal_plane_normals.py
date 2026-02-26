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
===========================
Crystal plane normals (hkl)
===========================

This example shows how to create crystal plane normals :math:`(hkl)`, visualize
them in the stereographic projection, and perform some simple operations.

A crystal plane normal :math:`(hkl)` is described by its normal vector
:math:`\mathbf{g} = h\cdot\mathbf{a^*} + k\cdot\mathbf{b^*} + l\cdot\mathbf{c^*}`, where
:math:`(\mathbf{a^*}, \mathbf{b^*}, \mathbf{c^*})` are the reciprocal crystal axes.
"""

# %%
from diffpy.structure import Lattice, Structure

from orix.crystal_map import Phase
from orix.plot import register_projections
from orix.vector import Miller

register_projections()

# %%
# To start with, let's create a tetragonal crystal with lattice parameters
# :math:`a` = :math:`b` = 0.5 nm and :math:`c` = 1 nm and symmetry elements described
# by point group *4*.

tetragonal = Phase(
    point_group="4",
    structure=Structure(lattice=Lattice(0.5, 0.5, 1, 90, 90, 90)),
)
print(tetragonal)
print(tetragonal.structure.lattice)

# %%
# Create two crystal plane normals and plot them in the stereographic projection.

g1 = Miller(hkl=[[1, 2, 0], [3, 1, 1]], phase=tetragonal)
print(g1)

g1.scatter(c=["b", "r"], axes_labels=["e1", "e2"], grid=True, grid_resolution=(90, 90))

# %%
# Let's compute the dot product in inverse nanometres and the angle in degrees between
# the vectors.

g120, g311 = g1

print(g120.dot(g311))
print(g120.angle_with(g311, degrees=True))

# %%
# We can get the reciprocal components of the lattice vector :math:`\mathbf{t} = [114]`
# (i.e. which lattice plane normal the [114] direction is parallel to) by accessing
# :attr:`~orix.vector.Miller.hkl`

print(Miller(uvw=[1, 1, 4], phase=tetragonal).hkl)

# %%
# We see that the lattice vector :math:`\mathbf{t} = [114]` is perpendicular to the
# lattice plane normal :math:`\mathbf{g} = (1 1 16)`.
#
# The length of reciprocal lattice vectors can also be accessed via
# :prop:`~orix.vector.Miller.length`.
# The length equals :math:`|\mathbf{g}| = \frac{1}{d_{\mathrm{hkl}}}` in reciprocal
# lattice parameter units (nm^-1 in this case)

print(g1.length)

# %%
# We can then obtain the interplanar spacing :math:`d_{\mathrm{hkl}}`

print(1 / g1.length)
