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
"""
=========
Wulff net
=========

Some examples for how a Wulff net can be added to any stereographic plot in
ORIX.
"""

import matplotlib.collections as mcollctions
import matplotlib.pyplot as plt
import numpy as np

from orix.plot import StereographicPlot
from orix.quaternion.symmetry import C6
from orix.vector.vector3d import Vector3d

fig, ax = plt.subplots(
    figsize=[6, 4], nrows=1, ncols=2, subplot_kw=dict(projection="stereographic")
)

# display a standard Wulff net with 2 degree minor markers and 10 degree
# major markers, with 10 degree caps at the tops and bottoms.
ax[0].wulff_net()

# The grid spacing can also be changed or turned on and off, similar to ax.grid
# in matplotlib. Here, the latitudinal grid is set to 3 degrees (15 degrees for
# major grid lines), and the longitudinal to 9 degrees (45 degrees for major grid
# lines)

ax[1].wulff_net(True, 3, 9, 15, 45, 15)
# Turn it off
ax[0].wulff_net()
# Then turn it back on, with the previously defined grid spacing saved.
ax[0].wulff_net()
ax[0].set_title("Standard")
ax[1].set_title("Custom")
plt.tight_layout()

# This also works for subsections, such as fundamental sectors.
fig = plt.figure()
ax = fig.add_subplot(projection="stereographic")
ax.restrict_to_sector(C6.fundamental_sector)
ax.wulff_net()
# Add some manual labels. the XY coordinates can be set by using either a single
# vector, or a polar-azimuth pair. Both methods are showb below for a 6-fold
# crystal.
ax.text(Vector3d([0, 0, 1]), s="[001]", offset=[0, -0.04], c="r")
ax.text(Vector3d([1, 0, 0]).unit, s="[100]", offset=[0, -0.04], c="g")
ax.text(np.pi / 3, np.pi / 2, s="[010]", offset=[-0, 0], c="b")
ax.set_title("C6 symmetry\n\n")
plt.tight_layout()
