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
========================
Crystal directions [uvw]
========================

This example shows how to create crystal lattice directions :math:`[uvw]`, visualize
them in the stereographic projection, and perform some simple operations.

A crystal lattice direction
:math:`\mathbf{t} = u\cdot\mathbf{a} + v\cdot\mathbf{b} + w\cdot\mathbf{c}` is a vector
with coordinates :math:`(u, v, w)` with respect to the crystal axes
:math:`(\mathbf{a}, \mathbf{b}, \mathbf{c})`, and is denoted :math:`[uvw]`.
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
# Create two lattice directions and plot them in the stereographic projection.

t1 = Miller(uvw=[[1, 2, 0], [3, 1, 1]], phase=tetragonal)
print(t1)

t1.scatter(c=["b", "r"], axes_labels=["e1", "e2"], grid=True, grid_resolution=(90, 90))

# %%
# Let's compute the dot product in nanometres and the angle in degrees between the
# vectors.

t120, t311 = t1

print(t120.dot(t311))
print(t120.angle_with(t311, degrees=True))

# %%
# We can get the length of a direct lattice vector :math:`|\mathbf{t}|`, given in
# lattice parameter units (nm in this case)

t2 = Miller(uvw=[0, -0.5, 0.5], phase=tetragonal)
print(t2.length)
