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
from diffpy.structure import Structure

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


def _calculate_basic_unit_cell_vertices(a1, a2, a3, alpha=90, beta=90, gamma=90):
    verts = np.array(list(product(*zip((0, 0, 0), (a1, a2, a3)))))
    center = verts.mean(axis=0)
    return verts - center  # center on (0, 0, 0)


def _calculate_basic_unit_cell_edges(verts, a1, a2, a3):
    verts = _calculate_basic_unit_cell_vertices(a1, a2, a3)
    # get valid edges from all unit cell egde possibilities unit cell
    edges_valid = [
        (s, e)
        for s, e in combinations(verts, 2)
        if np.isclose((a1, a2, a3), np.linalg.norm(s - e)).any()
    ]
    return np.array(edges_valid)


def _plot_unit_cell(rotation, c=None, axes_length=0.5, structure=None, **arrow_kwargs):
    # TODO: More than only cubic
    # introduce some basic non-cubic cell functionality

    if structure is None:
        a1, a2, a3 = 2, 2, 2
    else:
        # TODO: add some Structure support
        assert isinstance(structure, Structure)
        raise NotImplementedError

    verts = _calculate_basic_unit_cell_vertices(a1, a2, a3)
    edges = _calculate_basic_unit_cell_edges(verts, a1, a2, a3)
    edges_rotated = rotation * Vector3d(edges)

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.axis("off")
    ax.set_box_aspect((1, 1, 1))  # equal aspect

    # xrange, yrange, zrange = np.ptp(verts, axis=0)
    xmax, ymax, zmax = np.max(np.abs(verts), axis=0)
    lim = max(xmax, ymax, zmax)

    pad = 1.5
    axlim = pad * lim
    ax.set_xlim(-axlim, axlim)
    ax.set_ylim(-axlim, axlim)
    ax.set_zlim(-axlim, axlim)
    ax.margins(0, 0, 0)

    # default projection to +x -> east/right, +y -> north/upwards, +z out-of-page
    ax.azim = -90
    ax.elev = 90 - 1e-6  # add small offset to avoid arrow3D projection errors

    # axes colors from ParaView and TomViz
    colors = ("tab:red", "yellow", "tab:green")
    labels = ("x", "y", "z")

    arrow_kwargs.setdefault("mutation_scale", 20)
    arrow_kwargs.setdefault("arrowstyle", "-")
    arrow_kwargs.setdefault("linewidth", 2)

    # add lab reference frame axes and labels
    for i in range(3):
        _data = np.full((3, 2), -1.4 * lim)  # less padding than axlim
        _data[i, 1] += axes_length
        _label = labels[i]
        arrow = Arrow3D(
            *_data,
            color=colors[i],
            label=f"Sample reference axes {_label}",
            **arrow_kwargs,
        )
        ax.add_artist(arrow)
        ax.text3D(
            *_data[:, 1], f"${_label}_s$", label=f"Sample reference axes label {_label}"
        )  # s for sample

    if c is None:
        c = "tab:blue"

    for i, (v1, v2) in enumerate(edges_rotated.data):
        ax.plot3D(*zip(v1, v2), c=c, label=f"Lattice edge {i}")

    # add crystal reference frame axes and labels
    for i, v in enumerate(Vector3d(np.eye(3))):
        # rotate vector
        v1 = (rotation * v).data.ravel() * axes_length
        v0r = rotation * Vector3d(verts[0])  # offset axes to sit on crystal origin
        _data = (np.zeros((3, 2)).T + v0r.data).T
        _data[:, 1] += v1
        _label = labels[i]
        arrow = Arrow3D(
            *_data,
            color=colors[i],
            label=f"Crystal reference axes {_label}",
            **arrow_kwargs,
        )
        ax.add_artist(arrow)
        ax.text3D(
            *_data[:, 1],
            f"${_label}_c$",
            label=f"Crystal reference axes label {_label}",
        )  # c for crystal

    return fig
