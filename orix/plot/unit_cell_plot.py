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


def _plot_unit_cell(rotation):
    # TODO: More than only cubic
    # TODO: Add reference frame axes
    d = [-1, 1]
    xlim, ylim, zlim = (max(d),) * 3

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.axis("off")
    ax.set_box_aspect((xlim, ylim, zlim))
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_zlim(-zlim, zlim)
    ax.margins(0, 0, 0)

    for s, e in combinations(np.array(list(product(d, d, d))), 2):
        if np.sum(np.abs(s - e)) == (d[1] - d[0]):
            vs = rotation * Vector3d(s)
            ve = rotation * Vector3d(e)
            ax.plot3D(*zip(vs.data.squeeze(), ve.data.squeeze()), c="b")

    return fig
