# Copyright 2018-2024 the orix developers
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

from diffpy.structure import Lattice, Structure
from matplotlib import pyplot as plt
import numpy as np
import pytest

from orix.plot._util import Arrow3D
from orix.plot.unit_cell_plot import (
    _calculate_basic_unit_cell_edges,
    _calculate_basic_unit_cell_vertices,
)
from orix.quaternion import Orientation


def test_unit_cell_plot_default():
    ori = Orientation.random()
    fig = ori.plot_unit_cell(return_figure=True)
    assert len(fig.axes) == 1
    axes = fig.axes[0]
    assert len(axes.lines) == 12  # 12 edges in orthorhombic unit cell
    # 6 Arrow3D -> 3 for both sample and crystal reference frames
    assert len(axes.patches) == 6
    # test default projection
    assert axes.azim == -90
    assert round(axes.elev) == 90
    plt.close("all")

    plt.close("all")


def test_unit_cell_plot_multiple_orientations_raises():
    ori = Orientation.random(2)
    with pytest.raises(ValueError, match="Can only plot a single unit cell"):
        ori.plot_unit_cell()
    plt.close("all")


def test_unit_cell_plot_orthorhombic():
    ori = Orientation.random()
    lattice = Lattice(1, 2, 3, 90, 90, 90)
    structure = Structure(lattice=lattice)
    _ = ori.plot_unit_cell(return_figure=True, structure=structure)


def test_unit_cell_plot_hexagonal():
    ori = Orientation.random()
    lattice = Lattice(1, 1, 2, 90, 90, 120)
    structure = Structure(lattice=lattice)
    fig = ori.plot_unit_cell(return_figure=True, structure=structure)
    axes = fig.axes[0]
    # should only be 12 edges in hexagonal unit cell, this test checks
    # that the edges parallel to (0000)-(11-20) are not plotted
    assert len(axes.lines) == 12
    plt.close("all")


def test_unit_cell_plot_crystal_reference_axes_position_center():
    ori = Orientation.identity()
    a1, a2, a3 = 1, 1.5, 2
    lattice = Lattice(a1, a2, a3, 90, 90, 90)
    structure = Structure(lattice=lattice)
    # test cell center
    fig = ori.plot_unit_cell(
        return_figure=True,
        structure=structure,
        crystal_axes_loc="center",
    )
    arrows = fig.axes[0].patches
    crys_ref_ax = [p for p in arrows if "Crystal reference axes" in p.get_label()]
    crys_ref_ax_data = np.stack([np.array(a._verts3d) for a in crys_ref_ax])
    assert np.allclose(crys_ref_ax_data[:, :, 0], 0)
    plt.close("all")


def test_unit_cell_plot_crystal_reference_axes_position_origin():
    ori = Orientation.identity()
    a1, a2, a3 = 1, 1.5, 2
    lattice = Lattice(a1, a2, a3, 90, 90, 90)
    structure = Structure(lattice=lattice)
    # test cell center
    fig = ori.plot_unit_cell(
        return_figure=True,
        structure=structure,
        crystal_axes_loc="origin",
    )
    arrows = fig.axes[0].patches
    crys_ref_ax = [p for p in arrows if "Crystal reference axes" in p.get_label()]
    crys_ref_ax_data = np.stack([np.array(a._verts3d) for a in crys_ref_ax])
    assert np.allclose(crys_ref_ax_data[:, :, 0] + np.array((a1, a2, a3)) / 2, 0)
    plt.close("all")


def test_unit_cell_plot_crystal_reference_axes_position_raises():
    ori = Orientation.identity()
    with pytest.raises(ValueError, match="Crystal_axes_loc must be either"):
        ori.plot_unit_cell(crystal_axes_loc="test")
    plt.close("all")


def test_calculate_basic_unit_cell_raises():
    lattice = Lattice()
    with pytest.raises(ValueError, match=r"Vectors must be \(3, 3\) array."):
        _ = _calculate_basic_unit_cell_edges(lattice.base, np.ones((3, 4)))

    with pytest.raises(ValueError, match=r"Vectors must be \(3, 3\) array."):
        _ = _calculate_basic_unit_cell_vertices(np.ones((3, 4)))


def test_unit_cell_plot_invalid_structure_raises():
    ori = Orientation.random()
    with pytest.raises(TypeError, match=r"Structure must be diffpy.structure."):
        ori.plot_unit_cell(structure=np.arange(3))


def test_arrow3D():
    _, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    a = Arrow3D((0, 1), (0, 1), (0, 1), arrowstyle="-|>", mutation_scale=20)
    ax.add_artist(a)
    plt.draw()
    plt.close("all")
