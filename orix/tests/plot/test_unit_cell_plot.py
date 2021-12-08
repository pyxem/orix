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

import pytest

from diffpy.structure import Structure, Lattice
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
