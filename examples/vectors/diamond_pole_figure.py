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
A diamond [111] pole figure
===========================

Let's make a pole figure in the :math:`\mathbf{t}` = [111] direction of the diamond
structure, as seen in `this figure from Wikipedia
<https://commons.wikimedia.org/wiki/File:DiamondPoleFigure111.png>`__.

The figure caption reads as follows

> The spots in the stereographic projection show the orientation of lattice planes with
> the 111 in the center.
> Only poles for a non-forbidden Bragg reflection are shown between Miller indices -10
> <= (h,k,l) <= 10.
> The green spots contain Miller indices up to 3, for example 111, 113, 133, 200 etc in
> its fundamental order.
> Red are those raising to 5, ex. 115, 135, 335 etc, while blue are all remaining until
> 10, such as 119, 779, 10.10.00 etc.
"""

# %%
import numpy as np

from orix.crystal_map import Phase
from orix.plot import register_projections
from orix.quaternion import Rotation
from orix.vector import Miller, Vector3d

register_projections()

# %%

diamond = Phase(space_group=227)
t1 = Miller.from_highest_indices(phase=diamond, uvw=[10, 10, 10])
print(t1)

# %%
# Remove duplicates under symmetry using :math:`orix.vector.Miller.unique`.

t2 = t1.unique(use_symmetry=True)
print(t2.size)

# %%
# Symmetrise to get all symmetrically equivalent directions.

t3 = t2.symmetrise(unique=True)
print(t3)

# %%
# Remove forbidden reflections in face-centered cubic structures (all hkl must be all
# even or all odd).

selection = np.sum(np.mod(t3.hkl, 2), axis=1)
allowed = np.array([i not in [1, 2] for i in selection], dtype=bool)
t4 = t3[allowed]
print(t4)

# %%
# Assign colors to each class of vectors as per the description on Wikipedia.

uvw = np.abs(t4.uvw)
green = np.all(uvw <= 3, axis=-1)
red = np.any(uvw > 3, axis=-1) * np.all(uvw <= 5, axis=-1)
blue = np.any(uvw > 5, axis=-1)
rgb_mask = np.column_stack([red, green, blue])

# Sanity check
print(np.count_nonzero(rgb_mask) == t4.size)

# %%
# Rotate directions so that [111] impinges the unit sphere in the north pole (out of
# plane direction).

vz = Vector3d.zvector()
v111 = Vector3d([1, 1, 1])
R1 = Rotation.from_axes_angles(vz.cross(v111), -vz.angle_with(v111))
R2 = Rotation.from_axes_angles(vz, -15, degrees=True)
t5 = R2 * R1 * t4

# %%
# Restrict to upper hemisphere and remove duplicates.

is_upper = t5.z > 0
t6 = t5[is_upper]
rgb_mask2 = rgb_mask[is_upper]

_, idx = t6.unit.unique(return_index=True)
t7 = t6[idx]
rgb_mask2 = rgb_mask2[idx]

# %%
# Finally, plot the vectors.

rgb = np.zeros_like(t7.uvw)
rgb[rgb_mask2] = 1

t7.scatter(c=rgb, s=10, grid=False, figure_kwargs=dict(figsize=(12, 12)))
