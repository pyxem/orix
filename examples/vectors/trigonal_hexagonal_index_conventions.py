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
========================================
Trigonal and hexagonal index conventions
========================================

Crystal lattice vectors :math:`[uvw]` and plane normals :math:`(hkl)` in lattices with
trigonal and hexagonal crystal symmetry are typically expressed in Weber symbols
:math:`\mathbf{t} = [UVTW]` and Miller-Bravais indices :math:`\mathbf{g} = (hkil)`.

The definition of :math:`[UVTW]` used in orix follows *Introduction to Conventional
Transmission Electron Microscopy* (DeGraef, 2003)

.. math::

    U &= \frac{2u - v}{3},\\
    V &= \frac{2v - u}{3},\\
    T &= -\frac{u + v}{3},\\
    W &= w.

For a plane, the :math:`(h, k, l)` indices are the same in :math`(hkl)` and
:math:`(hkil)`, and :math:`i = -(h + k)`.

The first three Miller indices always sum up to zero, i.e.

.. math::

    U + V + T &= 0,\\
    h + k + i &= 0.
"""

# %%
from diffpy.structure import Lattice, Structure

from orix.crystal_map import Phase
from orix.plot import register_projections
from orix.vector import Miller

register_projections()

# %%
# Let's create a trigonal crystal.

trigonal = Phase(
    point_group="321",
    structure=Structure(lattice=Lattice(4.9, 4.9, 5.4, 90, 90, 120)),
)
print(trigonal)

# %%
t1 = Miller(UVTW=[2, 1, -3, 1], phase=trigonal)
print(t1)

# %%
g1 = Miller(hkil=[1, 1, -2, 3], phase=trigonal)
print(g1)

# %%
fig = t1.scatter(
    c="C0",
    grid=True,
    grid_resolution=(30, 30),
    axes_labels=["e1", "e2"],
    return_figure=True,
)
g1.scatter(figure=fig, c="C1")

# %%
# We can switch between the coordinate format of a vector.
# However, this does not change the vector, since all vectors are stored with respect to
# the Cartesian coordinate system internally.

print(g1, "\n\n", g1.data)

g1.coordinate_format = "UVTW"
print(g1, "\n\n", g1.data)

# %%
# Getting the closest integer indices, however, changes the vector.

g2 = g1.round()
print(g2, "\n\n", g2.data)
