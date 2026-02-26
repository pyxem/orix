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
=========================================
Directions and planes in rotated crystals
=========================================

Let's consider the orientation of a cubic crystal rotated :math:`45^{\circ}` about the
sample :math:`\mathbf{Z}` axis.
This orientation is denoted :math:`(\hat{\mathbf{n}}, \omega) = ([001], -45^{\circ})`
in axis-angle notation (see discussion by :cite:t:`rowenhorst2015consistent`).
Orientations in orix are *interpreted* as basis transformations from the sample to the
crystal (so-called "lab2crystal").
We therefore have to invert the orientation to get a crystal direction or plane normal
expressed in the sample reference frame.
"""

# %%
from diffpy.structure import Lattice, Structure
import numpy as np

from orix.crystal_map import Phase
from orix.plot import register_projections
from orix.quaternion import Orientation
from orix.vector import Miller, Vector3d

register_projections()

# %%
# Create the cubic crystal phase.

cubic = Phase(point_group="m-3m")
print(cubic, "\n", cubic.structure.lattice.abcABG())

# %%
# Create the orientation.
O1 = Orientation.from_axes_angles([0, 0, 1], -45, cubic.point_group, degrees=True)
print(O1)

# %%
# We can apply this orientation to a crystal direction :math:`\mathbf{t} = [uvw]` to
# find this direction in this particular crystal with respect to the sample coordinate
# system, denoted :math:`\mathbf{v} = O^{-1} \cdot \mathbf{t}`.

t1 = Miller(uvw=[1, 1, 1], phase=cubic)
v1 = Vector3d(~O1 * t1)
print(v1)

# %%
# [uvw] in unrotated crystal with orientation the identity orientation
fig = t1.scatter(c="r", return_figure=True, axes_labels=["X", "Y"])

# [uvw] in rotated crystal with (n, omega) = ([001], -45 deg)
(~O1 * t1).scatter(figure=fig, c="b", marker="s")

# %%
# We see that the :math:`[111]` vector in the rotated crystal with orientation
# :math:`O = (\hat{\mathbf{n}}, \omega) = ([001], -45^{\circ})` lies in the sample
# Y-Z plane.

# We can apply all cubic crystal symmetry operations :math:`s_i` to obtain the
# coordinates with respect to the sample reference frame for all crystallographically
# equivalent, but unique, directions,
# :math:`\mathbf{v} = O^{-1} \cdot (s_i \cdot \mathbf{t} \cdot s_i^{-1})`.

(~O1 * t1.symmetrise(unique=True)).scatter(
    c="b", marker="s", axes_labels=["X", "Y"], hemisphere="both"
)

# %%
# The same applied to a trigonal crystal direction.

trigonal = Phase(
    point_group="321",
    structure=Structure(lattice=Lattice(4.9, 4.9, 5.4, 90, 90, 120)),
)
print(trigonal)

O2 = Orientation.from_euler([10, 20, 30], trigonal.point_group, degrees=True)
print(O2)

g1 = Miller(hkil=[1, -1, 0, 0], phase=trigonal)
print(g1)

v2 = ~O2 * g1.symmetrise(unique=True)
v2.scatter(
    hemisphere="both",
    grid_resolution=(30, 30),
    figure_kwargs=dict(figsize=(10, 5)),
    axes_labels=["X", "Y"],
)

# %%
# The stereographic plots above are essentially the :math:`\mathbf{g} = \{1\bar{1}00\}`
# pole figure representation of the orientation :math:`O_2`.
