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
==============================
Restrict to fundamental sector
==============================

This example shows how to restrict the stereographic plot to the fundamental sector of
one of the eleven Laue group symmetries :math:`S` using
:meth:`~orix.plot.StereographicPlot.restrict_to_sector`.
The sector is typically obtained from
:attr:`orix.quaternion.Symmetry.fundamental_sector`.
It is often called the 'fundamental triangle' or
'standard stereographic triangle (SST)'.

We demonstrate this functionality by drawing (near) great circles about some typically
strongly reflecting low-index reciprocal lattice vectors :math:`\mathbf{g} = \{hkl\}` in
crystals of point group :math:`S = m\bar{3}m`.
The deviations from the great circles are related to the kinematically calculated width
of a Kikuchi band scattered from these vectors, assuming a lattice parameter of
$a$ = 0.404 nm (aluminium) and an accelerating voltage of 20 kV.
The band width is assumed to be two times the Bragg angle :math:`\theta`.
"""

import matplotlib.pyplot as plt
import numpy as np

from orix.crystal_map import Phase
from orix.plot import format_labels, register_projections
from orix.quaternion import symmetry
from orix.vector import Miller

plt.rcParams["font.size"] = 15

register_projections()  # Register our custom Matplotlib projections

# Symmetrically equivalent set of hkl
g0 = Miller(
    hkl=[[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]],
    phase=Phase(point_group=symmetry.Oh),
)
print(g0)
g, idx = g0.symmetrise(unique=True, return_index=True)

# Width of Kikuchi bands (deviation from great circles)
theta = np.deg2rad([1.054, 1.218, 1.722, 2.019])
theta = theta[idx]

# Plot pair of near great circles
fig = g.draw_circle(opening_angle=np.pi / 2 + theta, return_figure=True)
g.draw_circle(opening_angle=np.pi / 2 - theta, figure=fig)

# Restrict to fundamental sector of m-3m (with some padding outside sector)
ax = fig.axes[0]
ax.restrict_to_sector(
    g.phase.point_group.fundamental_sector, edgecolor="r", lw=2, pad=5
)

# Get symmetrically equivalent set of zone axes t = <uvw>
t = g.reshape(g.size, 1).cross(g.reshape(1, g.size)).flatten()
t = t.in_fundamental_sector()
t = t.unique(use_symmetry=True)
t = t.round().unique()
t.scatter(
    figure=fig,
    c="none",
    vector_labels=format_labels(t.coordinates),
    text_kwargs={"va": "center", "bbox": {"fc": "w", "pad": 1, "alpha": 0.75}},
)

_ = ax.set_title(r"Low-index $[uvw]$ in fundamental sector of $m\bar{3}m$", pad=10)
