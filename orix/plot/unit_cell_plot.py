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

from diffpy.structure import Structure, Lattice
import matplotlib.pyplot as plt
import numpy as np

from orix.vector import Vector3d
from orix.plot._util import Arrow3D


def _calculate_basic_unit_cell_vertices(a1, a2, a3):
    """Calculate cell vertices for orthorhomic unit cells."""
    verts = np.array(list(product(*zip((0, 0, 0), (a1, a2, a3)))))
    center = verts.mean(axis=0)
    return verts - center  # center on (0, 0, 0)


def _calculate_basic_unit_cell_edges(verts, a1, a2, a3):
    """Calculate valid unit cell edges for orthorhombic until cells."""
    verts = _calculate_basic_unit_cell_vertices(a1, a2, a3)
    # get valid edges from all unit cell egde possibilities unit cell
    edges_valid = []
    # for all possible combinations of vertices, keep if the distance between them is
    # equal to any of the basis vectors
    for v1, v2 in combinations(verts, 2):
        if np.isclose((a1, a2, a3), np.linalg.norm(v2 - v1)).any():
            edges_valid.append((v1, v2))
    return np.array(edges_valid)


def _plot_unit_cell(
    rotation, c="tab:blue", axes_length=0.5, structure=None, **arrow_kwargs
):
    """Plot the unit cell orientation, showing the sample and crystal reference frames.

    Parameters
    ----------
    rotation : orix.quaternion.Rotation
        Rotation of the unit cell.
    c : str, optional
        Unit cell edge color.
    axes_length : float, optional
        Length of the reference axes in Angstroms, by default 0.5.
    structure : diffpy.structure.Structure or None, optional
        Structure of the unit cell, only orthorhombic lattices are currently
        supported. If not given, a cubic unit cell with a lattice parameter of 2
        will be plotted.
    arrow_kwargs : dict, optional
        Keyword arguments passed to
        :class:`matplotlib.patches.FancyArrowPatch`, for example"arrowstyle".
        Passed to matplotlib.patches.FancyArrowPatch, for example 'arrowstyle'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plotted figure.
    """
    # requires active rotation of the lattice in the sample reference frame
    rotation = ~rotation
    # TODO: More than only cubic
    # introduce some basic non-cubic cell functionality
    if structure is None:
        structure = Structure(lattice=Lattice(2, 2, 2, 90, 90, 90))

    assert isinstance(
        structure, Structure
    ), "Structure must be diffpy.structure.Structure."
    lattice = structure.lattice
    if not (lattice.alpha == lattice.beta == lattice.gamma == 90):
        raise ValueError("Only orthorhombic lattices are currently supported.")
    a1, a2, a3 = lattice.a, lattice.b, lattice.c

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
        # _s for sample
        ax.text3D(
            *_data[:, 1], f"${_label}_s$", label=f"Sample reference axes label {_label}"
        )

    for i, (v1, v2) in enumerate(edges_rotated.data):
        ax.plot3D(*zip(v1, v2), c=c, label=f"Lattice edge {i}")

    # add crystal reference frame axes and labels
    for i, v in enumerate(Vector3d(np.eye(3))):
        # rotate vector
        v1 = (rotation * v).data.ravel() * axes_length
        # setup verts for reference axes
        _data = np.zeros((3, 2))
        _data[:, 1] += v1
        # rotate cell origin into new position
        v0r = rotation * Vector3d(verts[0])
        # offset axes to sit on cell origin
        _data = (_data.T + v0r.data).T
        _label = labels[i]
        arrow = Arrow3D(
            *_data,
            color=colors[i],
            label=f"Crystal reference axes {_label}",
            **arrow_kwargs,
        )
        ax.add_artist(arrow)
        # _c for crystal
        ax.text3D(
            *_data[:, 1],
            f"${_label}_c$",
            label=f"Crystal reference axes label {_label}",
        )

    return fig
