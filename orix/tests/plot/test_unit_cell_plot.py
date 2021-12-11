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

from packaging import version

from diffpy.structure import Lattice, Structure
from matplotlib import __version__ as _MPL_VERSION
import numpy as np
import pytest

from orix.quaternion import Orientation


def test_unit_cell_plot_default():
    ori = Orientation.random()
    fig = ori.plot_unit_cell(return_figure=True)
    assert len(fig.axes) == 1
    axes = fig.axes[0]
    assert len(axes.lines) == 12  # 12 edges in orthorhombic unit cell
    # 6 Arrow3D -> 3 for both sample and crystal reference frames
    if version.parse(_MPL_VERSION) >= version.parse("3.4"):  # pragma: no cover
        assert len(axes.patches) == 6
    else:
        assert len(axes.artists) == 6
    # test default projection
    assert axes.azim == -90
    assert round(axes.elev) == 90


def test_unit_cell_plot_multiple_orientations_raises():
    ori = Orientation.random((2,))
    with pytest.raises(ValueError, match="Can only plot a single unit cell"):
        ori.plot_unit_cell()


def test_unit_cell_plot_orthorhombic():
    ori = Orientation.random()
    lattice = Lattice(1, 2, 3, 90, 90, 90)
    structure = Structure(lattice=lattice)
    fig = ori.plot_unit_cell(return_figure=True, structure=structure)


def test_unit_cell_plot_nonorthorhombic_raises():
    ori = Orientation.random()
    lattice = Lattice(1, 2, 3, 90, 91, 90)
    structure = Structure(lattice=lattice)
    with pytest.raises(ValueError, match="Only orthorhombic lattices"):
        fig = ori.plot_unit_cell(return_figure=True, structure=structure)


def test_unit_cell_plot_crystal_reference_axes_position_center():
    ori = Orientation.identity()
    lattice = Lattice(2, 2, 2, 90, 90, 90)
    structure = Structure(lattice=lattice)
    # test cell center
    fig = ori.plot_unit_cell(
        return_figure=True,
        structure=structure,
        crystal_reference_frame_axes_position="center",
    )
    if version.parse(_MPL_VERSION) >= version.parse("3.4"):  # pragma: no cover
        arrows = fig.axes[0].patches
    else:
        arrows = fig.axes[0].artists
    crys_ref_ax = [p for p in arrows if "Crystal reference axes" in p.get_label()]
    crys_ref_ax_data = np.concatenate([np.array(a._verts3d) for a in crys_ref_ax])
    assert np.allclose(crys_ref_ax_data[:, 0], 0)


def test_unit_cell_plot_crystal_reference_axes_position_origin():
    ori = Orientation.identity()
    lattice = Lattice(2, 2, 2, 90, 90, 90)
    structure = Structure(lattice=lattice)
    # test cell center
    fig = ori.plot_unit_cell(
        return_figure=True,
        structure=structure,
        crystal_reference_frame_axes_position="origin",
    )
    if version.parse(_MPL_VERSION) >= version.parse("3.4"):  # pragma: no cover
        arrows = fig.axes[0].patches
    else:
        arrows = fig.axes[0].artists
    crys_ref_ax = [p for p in arrows if "Crystal reference axes" in p.get_label()]
    crys_ref_ax_data = np.concatenate([np.array(a._verts3d) for a in crys_ref_ax])
    assert np.allclose(crys_ref_ax_data[:, 0], -1)
