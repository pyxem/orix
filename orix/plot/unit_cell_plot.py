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

from orix.plot._util import Arrow3D
from orix.vector import Vector3d


def _calculate_basic_unit_cell_vertices(vectors):
    """Calculate cell vertices for unit cells."""
    vectors = np.asarray(vectors)
    if vectors.shape != (3, 3):
        raise ValueError("Vectors must be (3, 3) array.")
    # generate list of lattice basis vectors from (000) to (111) (+ve)
    verts = np.array(list(product(*zip((0, 0, 0), (1, 1, 1)))))
    verts = (verts[..., np.newaxis] * vectors).sum(axis=1)
    center = verts.mean(axis=0)
    return verts - center  # center on (0, 0, 0)


def _calculate_basic_unit_cell_edges(verts, vectors):
    """Calculate valid unit cell edges for unit cells."""
    vectors = np.asarray(vectors)
    if vectors.shape != (3, 3):
        raise ValueError("Vectors must be (3, 3) array.")
    a1, a2, a3 = np.linalg.norm(vectors, axis=-1)
    # get valid edges from all unit cell edge possibilities unit cell
    edges_valid = []
    # for all possible combinations of vertices, keep if the distance
    # between them is equal to any of the basis vectors
    for v1, v2 in combinations(verts, 2):
        if np.isclose((a1, a2, a3), np.linalg.norm(v2 - v1)).any():
            # extra case for hexagonal unit cells, do not plot
            # (0000)-(11-20) edge
            # in hexagonal cells the gamma angle is 120 degrees
            # (between a1 and a2). If xy of (v2 - v1) is equal to xy of
            # vectors[0] + vectors[1] -> skip
            if np.isclose((v2 - v1)[:-1], (vectors[0] + vectors[1])[:-1]).all():
                continue
            edges_valid.append((v1, v2))
    return np.array(edges_valid)


def _plot_unit_cell(
    rotation,
    c="tab:blue",
    axes_length=0.5,
    structure=None,
    crystal_axes_loc="origin",
    **arrow_kwargs,
):
    """Plot the unit cell orientation, showing the sample and crystal
    reference frames.

    Parameters
    ----------
    rotation : orix.quaternion.Rotation
        Rotation of the unit cell.
    c : str, optional
        Unit cell edge color.
    axes_length : float, optional
        Length of the reference axes in Angstroms, by default 0.5.
    structure : diffpy.structure.Structure or None, optional
        Structure of the unit cell, only orthorhombic lattices are
        currently supported. If not given, a cubic unit cell with a
        lattice parameter of 2 Angstroms will be plotted.
    crystal_axes_loc : str, optional
        Plot the crystal reference frame axes at the "origin" (default)
        or "center" of the plotted cell.
    arrow_kwargs : dict, optional
        Keyword arguments passed to
        :class:`matplotlib.patches.FancyArrowPatch`, for example
        `arrowstyle`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The plotted figure.
    """
    # active rotation of the lattice in the sample reference frame
    inv_rotation = ~rotation

    crystal_axes_loc = crystal_axes_loc.lower()
    if crystal_axes_loc not in ("origin", "center"):
        raise ValueError('Crystal_axes_loc must be either "origin" or "center".')

    # TODO: Introduce some basic non-cubic cell functionality
    if structure is None:
        structure = Structure(lattice=Lattice(2, 2, 2, 90, 90, 90))
    elif not isinstance(structure, Structure):
        raise TypeError("Structure must be diffpy.structure.Structure.")
    lattice = structure.lattice
    lattice_vectors = lattice.base

    verts = _calculate_basic_unit_cell_vertices(lattice_vectors)
    edges = _calculate_basic_unit_cell_edges(verts, lattice_vectors)
    edges_rotated = inv_rotation * Vector3d(edges)

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.axis("off")

    xmax, ymax, zmax = np.max(np.abs(verts), axis=0)
    lim = max(xmax, ymax, zmax)

    pad = 1.5
    axlim = pad * lim
    ax.set_xlim(-axlim, axlim)
    ax.set_ylim(-axlim, axlim)
    ax.set_zlim(-axlim, axlim)
    ax.set_box_aspect((1, 1, 1))  # equal aspect
    ax.margins(0, 0, 0)

    # default projection to +x -> east/right, +y -> north/upwards,
    # +z out-of-page
    ax.azim = -90
    # add small offset to avoid arrow3D projection errors
    ax.elev = 90 - 1e-6

    # axes colors from ParaView and TomViz
    colors = ("tab:red", "yellow", "tab:green")
    labels = ("x", "y", "z")

    arrow_kwargs.setdefault("mutation_scale", 20)
    arrow_kwargs.setdefault("arrowstyle", "-")
    arrow_kwargs.setdefault("linewidth", 2)

    # add lab reference frame axes and labels
    for i in range(3):
        data = np.full((3, 2), -1.4 * lim)  # less padding than axlim
        data[i, 1] += axes_length
        label = labels[i]
        arrow = Arrow3D(
            *data,
            color=colors[i],
            label=f"Sample reference axes {label}",
            **arrow_kwargs,
        )
        ax.add_artist(arrow)
        # _s for sample
        ax.text3D(
            *data[:, 1], f"${label}_s$", label=f"Sample reference axes label {label}"
        )

    for i, (v1, v2) in enumerate(edges_rotated.data):
        ax.plot3D(*zip(v1, v2), c=c, label=f"Lattice edge {i}")

    # add crystal reference frame axes and labels
    v_ref_ax = inv_rotation * Vector3d(np.eye(3))
    if crystal_axes_loc == "origin":  # cell origin
        crys_ref_ax_origin = Vector3d(verts[0])
    else:
        crys_ref_ax_origin = Vector3d.zero((1,))
    # rotate cell origin into new position
    cell_origin_rotated = inv_rotation * crys_ref_ax_origin

    for i, v in enumerate(v_ref_ax):
        # setup verts for reference axes
        data = np.zeros((3, 2))
        data[:, 1] = v.data.ravel() * axes_length
        # offset axes to sit on cell origin
        data = (data.T + cell_origin_rotated.data).T
        label = labels[i]
        arrow = Arrow3D(
            *data,
            color=colors[i],
            label=f"Crystal reference axes {label}",
            **arrow_kwargs,
        )
        ax.add_artist(arrow)
        # _c for crystal
        ax.text3D(
            *data[:, 1], f"${label}_c$", label=f"Crystal reference axes label {label}"
        )

    return fig
