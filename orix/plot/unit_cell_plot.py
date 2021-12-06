# -*- coding: utf-8 -*-
# Copyright 2018-2021 the orix developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix.  If not, see <http://www.gnu.org/licenses/>.

"""Private tools for plotting unit cells given a crystal symmetry and
orientation.
"""

from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np

from orix.vector import Vector3d

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np

from orix.vector import Vector3d

# taken from SO post https://stackoverflow.com/a/22867877/12063126
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        # no in plane projection, ie. vector is vertical
        # if np.allclose(np.flatten((xs, ys)), 0):
        #     self.axes.plot3D(0, 0, 0, m="o")
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        zorder = 1
        return zorder

    # matplotlib>=3.5 compatibility
    do_3d_projection = draw


def _plot_unit_cell(rotation, c=None, axes_length=0.5, **arrow_kwargs):
    # TODO: More than only cubic
    d = [-1, 1]
    xlim, ylim, zlim = (max(d),) * 3

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.axis("off")
    ax.set_box_aspect((xlim, ylim, zlim))

    offset = -1.5
    ax.set_xlim(offset, -offset)
    ax.set_ylim(offset, -offset)
    ax.set_zlim(offset, -offset)
    ax.margins(0, 0, 0)

    # default projection to +x -> east/right, +y -> north/upwards, +z out-of-page
    ax.azim = -90
    ax.elev = 90 - 1e-6  # add small offset to avoid arrow3D projection errors

    # axes colors from ParaView and TomViz
    colors = ("tab:red", "yellow", "tab:green")
    labels = ("x", "y", "z")

    arrow_kwargs.setdefault("mutation_scale", 20)
    arrow_kwargs.setdefault("arrowstyle", "-")

    # add lab reference frame axes and labels
    for i in range(3):
        _data = np.full((3, 2), offset)
        _data[i, 1] += axes_length

        arrow = Arrow3D(*_data, color=colors[i], **arrow_kwargs)
        ax.add_artist(arrow)

        ax.text3D(*_data[:, 1], f"${labels[i]}_s$")  # s for sample

    # add crystal reference frame axes and labels
    for i, v in enumerate(Vector3d(np.eye(3))):
        # rotate vector
        v1 = (rotation * v * axes_length).data.ravel()
        arrow = Arrow3D(
            (0, v1[0]), (0, v1[1]), (0, v1[2]), color=colors[i], **arrow_kwargs
        )
        ax.add_artist(arrow)
        ax.text3D(*v1, f"${labels[i]}_c$")  # c for crystal

    if c is None:
        c = "tab:blue"

    for s, e in combinations(np.array(list(product(d, d, d))), 2):
        if np.sum(np.abs(s - e)) == (d[1] - d[0]):
            vs = rotation * Vector3d(s)
            ve = rotation * Vector3d(e)
            ax.plot3D(*zip(vs.data.squeeze(), ve.data.squeeze()), c=c)

    return fig
