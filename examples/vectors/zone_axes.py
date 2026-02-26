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
=========
Zone axes
=========

This example shows how to use the :class:`~orix.vector.Miller` to work with zone axes.
"""

# %%
from diffpy.structure import Lattice, Structure

from orix.crystal_map import Phase
from orix.plot import register_projections
from orix.vector import Miller

register_projections()

# %%
# Let's use a tetragonal crystal.

tetragonal = Phase(
    point_group="4",
    structure=Structure(lattice=Lattice(0.5, 0.5, 1, 90, 90, 90)),
)
print(tetragonal)
print(tetragonal.structure.lattice)

# %%
# The cross product :math:`\mathbf{g} = \mathbf{t}_1 \cross \mathbf{t}_2` of the lattice
# vectors :math:`\mathbf{t}_1 = [110]` and :math:`\mathbf{t}_2 = [111]` in a tetragonal,
# crystal in direct space, is described by a vector expressed in reciprocal space,
# :math:`(hkl)`.

t1 = Miller(uvw=[[1, 1, 0], [1, 1, 1]], phase=tetragonal)
g1 = t1[0].cross(t1[1])
print(g1)

# %%
# The exact "indices" are returned.
# However, we can get new vectors with the indices rounded down or up to the *closest*
# smallest integer with :meth:`~orix.vector.Miller.round`.

print(g1.round())

# %%
# The maximum index that returned can be set by passing the ``max_index`` parameter.
#
# We can plot these direct lattice vectors :math:`[uvw]` and the vectors normal to the
# cross product vector, i.e. normal to the reciprocal lattice vector :math:`(hkl)`.

fig = t1.scatter(
    return_figure=True,
    c="r",
    axes_labels=["e1", "e2"],
    grid=True,
    grid_resolution=(90, 90),
)
g1.draw_circle(figure=fig, color="b", linestyle="--")

# %%
# Likewise, the cross product of reciprocal lattice vectors :math:`\mathbf{g}_1 = (110)`
# and :math:`\mathbf{g}_2 = (111)` results in a direct lattice vector.


g2 = Miller(hkl=t1.uvw, phase=tetragonal)
t2 = g2[0].cross(g2[1]).round()
print(t2)

fig = g2.scatter(
    return_figure=True,
    c="b",
    axes_labels=["e1", "e2"],
    grid=True,
    grid_resolution=(90, 90),
)
t2.draw_circle(figure=fig, color="r", linestyle="--")
