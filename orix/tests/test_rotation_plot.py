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

from matplotlib import pyplot as plt
import numpy as np

from orix.plot import RodriguesPlot, AxAnglePlot
from orix.quaternion import Misorientation, OrientationRegion
from orix.quaternion.symmetry import C1, D6


def test_init_rodrigues_plot():
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(auto_add_to_figure=False, projection="rodrigues")
    assert isinstance(ax, RodriguesPlot)


def test_init_axangle_plot():
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(auto_add_to_figure=False, projection="axangle")
    assert isinstance(ax, AxAnglePlot)


def test_RotationPlot_methods():
    """This code is lifted from demo-3-v0.1."""
    misori = Misorientation([1, 1, 1, 1])  # any will do
    fig = plt.figure()
    ax = fig.add_subplot(
        auto_add_to_figure=False, projection="axangle", proj_type="ortho"
    )
    ax.scatter(misori)
    ax.plot(misori)
    ax.plot_wireframe(OrientationRegion.from_symmetry(D6, D6))
    plt.close("all")

    # Clear the edge case
    ax.transform(np.asarray([1, 1, 1]))


def test_full_region_plot():
    empty = OrientationRegion.from_symmetry(C1, C1)
    _ = empty.get_plot_data()
