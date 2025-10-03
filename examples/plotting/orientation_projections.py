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
=======================
Orientation projections
=======================

This example demonstrates the four different projections used by ORX to
project Orientations (which are non-Euclidean) into either 2D or 3D
orthogonal space. This is done by subclassing matplotlib's Axes and
Axes3D classes.

There are four options for plotting projections. The first and only 2D
option is :class:`~orix.plot.InversePoleFigurePlot`, which is calculated
as  :math:`(X, Y) = ((v_x/(1-v_z)),(v_y/(1-v_z)))`. This is computationally
efficient and translates well to print publication, but loses orientation
information perpendicular to whatever axis is being plotted, similar to a 2D
X-ray of a skeleton.

The next three are 3D axis-angle projections, sometimes also called
Neo-Eulerian projections. Here, the fact that any rotation can be described
by a twist math:`\omega` around an axis :math:`\hat{\mathbf{n}}`. The
math:`(X,Y,Z)` coordinates of the orientation's projection is the
math:`(V_x,V_y,V_z)` coordinates of a unit vector describing
math:`\hat{\mathbf{n}}`, scaled by a function of :math:`\omega`.
The scaling options are:

    * :class:`~orix.plot.AxAnglePlot`:
        math:`\omega * \hat{\mathbf{n}}`, A linear projection
    * :class:`~orix.plot.RodriguesPlot`
        math:`tan(\omega/2) * \hat{\mathbf{n}}`, A Rectilinear projection,
        where orientations sharing a common rotation axis linearly align
    * :class:`~orix.plot.HomochoricPlot`
        math:`(0.75*(\omega-sin(\omega)))^{1/3} * \hat{\mathbf{n}}`, An
        equal-volume projection, where a cube placed anywhere inside takes
        up an identical solid angle in orientation space.

Note this list is not exhaustive and the descriptions are simplified.
For a deeper dive into the advantages and disadganvages of these projections
as well as enlightening comparisions of their warping of orientation space,
refer to the following open access publication:


.. _On three-dimensional misorientation spaces: https://royalsocietypublishing.org/doi/10.1098/rspa.2017.0274

(doi link: https://doi.org/10.1098/rspa.2017.0274)

"""

import matplotlib.pyplot as plt
import numpy as np

from orix.plot import IPFColorKeyTSL, register_projections
from orix.quaternion import Orientation, OrientationRegion
from orix.quaternion.symmetry import D3

plt.close("all")
register_projections()  # Register our custom Matplotlib projections
np.random.seed(2319)  # Create reproducible random data

n = 30
ori = Orientation.random(n, symmetry=D3)
# create orientation-dependent colormap for more informative plots.
color_key = IPFColorKeyTSL(D3)
clrs = color_key.orientation2color(ori)

############################################################################
# Orientation plots can be made in one of two ways. The first and simplest
# is via Orientation.scatter().
fig = plt.figure(figsize=(12, 3), layout="constrained")
ori.scatter(c=clrs, position=(1, 4, 1), projection="axangle", figure=fig)
fig.axes[0].set_title("Axis-Angle Projection")
ori.scatter(c=clrs, position=(1, 4, 2), projection="rodrigues", figure=fig)
fig.axes[1].set_title("Rodrigues Projection")
ori.scatter(c=clrs, position=(1, 4, 3), projection="homochoric", figure=fig)
fig.axes[2].set_title("Homochoric Projection")
# TODO: Following line does not plot properly due to the logic in
# Orientation.scatter
# ori.scatter(c=clrs, position=(1, 4, 4), projection="ipf", figure=fig)
plt.tight_layout()


# This can also be used to create standalone figures
ori.scatter(c=clrs, projection="ipf")
plt.tight_layout()


############################################################################
# The second method is by setting the projections when defining the
# matplotlib axes. This can require more tinkering since the plots are not
# auto-formatted like above, but it allows for more customization as well
# as the plotting of multiple datasets on a single plot

fig = plt.figure(figsize=(12, 4), layout="constrained")
ax_ax = fig.add_subplot(1, 4, 1, projection="axangle")
ax_rod = fig.add_subplot(1, 4, 2, projection="rodrigues")
ax_hom = fig.add_subplot(1, 4, 3, projection="homochoric")
ax_ipf = fig.add_subplot(1, 4, 4, projection="ipf", symmetry=D3)

ax_ipf.scatter(ori, c=clrs)

ax_ax.set_title("Axis-Angle Projection")
ax_rod.set_title("Rodrigues Projection")
ax_hom.set_title("Homochoric Projection")
ax_ipf.set_title("Inverse Pole Figure Projection \n\n")

fundamental_zone = OrientationRegion.from_symmetry(ori.symmetry)
for ax in [ax_ax, ax_rod, ax_hom]:
    ax.plot_wireframe(fundamental_zone)
    ax.set_proj_type = "ortho"
    ax.axis("off")
    ax.scatter(ori, c=clrs)

plt.tight_layout()
