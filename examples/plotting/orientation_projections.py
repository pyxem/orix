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

This example demonstrates how ORIX can be used to plot orientations.

Plotting orientations requires defining some projection between
orientation space and Euclidean space, which will by necessity
introduce distortion. This problem is similar to how any 2D map
of Earth's surface will always have a spatially-dependent scalebar.

ORIX currently includes four common projections, three Neo-Eulerian projections for
3D plots and one Stereographic projection for 2D plots. All four begin with the
axis-angle representation of an orientation as an angular rotation :math:`\omega` around a
unit vector axis :math:`\hat{\mathbf{n}}`.

    - :class:`~orix.plot.AxAnglePlot` :math:`(X,Y,Z) = \omega * \hat{\mathbf{n}}`, A linear Neo-Eulerian projection.
    - :class:`~orix.plot.RodriguesPlot` :math:`(X,Y,Z) = tan(\omega/2) * \hat{\mathbf{n}}`,  A Rectilinear Neo-Eulerian projection, where orientations sharing a common rotation axis are linearly aligned.
    - :class:`~orix.plot.HomochoricPlot` :math:`(X,Y,Z) = (0.75*(\omega-sin(\omega)))^{1/3} * \hat{\mathbf{n}}`, An equal-volume Neo-Eulerian projection, where a cube placed anywhere inside takes up an identical solid angle in orientation space.
    - :class:`~orix.plot.InversePoleFigurePlot` :math:`(X,Y) = ((\hat{\mathbf{n}_x}/(1-\hat{\mathbf{n}_z})),(\hat{\mathbf{n}_y}/(1-\hat{\mathbf{n}_z})))` : An angle-preserving stereogrpahic plot.

More information on these and other projections can be found in the
open-access publication "On three-dimensional misorientation spaces"
(https://doi.org/10.1098/rspa.2017.0274).

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
ori.scatter(c=clrs, position=(1, 4, 4), projection="ipf", figure=fig)
fig.axes[3].set_title("Inverse Pole Figure Projection \n\n")

plt.tight_layout()

############################################################################
# This can also be used to create standalone figures
ori.scatter(c=clrs, projection="ipf")

plt.tight_layout()

############################################################################
# The second method is by setting the projections when defining the
# matplotlib axes. This can require more tinkering since the plots are not
# auto-formatted like above, but it allows for more customization as well
# as the plotting of multiple datasets on a single plot

# Additionally, for these plots we will set the scaling equivalent in order
# to better illustrate the differences in the scales of the
# Neo-Eulerian plotting methods.
fig = plt.figure(figsize=(12, 3), layout="constrained")
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
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.scatter(ori, c=clrs)

plt.tight_layout()
