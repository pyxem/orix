# -*- coding: utf-8 -*-
# Copyright 2018-2020 the orix developers
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

from orix.plot.rotation_plot import RodriguesPlot, AxAnglePlot
from orix.quaternion.orientation import Misorientation
from orix.quaternion.symmetry import D6, C1
from orix.quaternion.orientation_region import OrientationRegion


def test_init_RodriguesPlot():
    fig = plt.figure(figsize=(3, 3))
    _ = RodriguesPlot(fig)
    return None


def test_init_AxAnglePlot():
    fig = plt.figure(figsize=(3, 3))
    _ = AxAnglePlot(fig)
    return None


def test_RotationPlot_methods():
    """ This code is lifted from demo-3-v0.1 """
    misori = Misorientation([1, 1, 1, 1])  # any will do
    fig = plt.figure(figsize=(6, 3))
    gridspec = plt.GridSpec(1, 1, left=0, right=1, bottom=0, top=1, hspace=0.05)
    ax_misori = fig.add_subplot(
        gridspec[0], projection="axangle", proj_type="ortho", aspect="auto"
    )
    ax_misori.scatter(misori)
    ax_misori.plot(misori)
    ax_misori.plot_wireframe(OrientationRegion.from_symmetry(D6, D6))
    plt.close("all")

    # clear the edge case
    ax_misori.transform(np.asarray([1, 1, 1]))
    return None


def test_full_region_plot():
    empty = OrientationRegion.from_symmetry(C1, C1)
    _ = empty.get_plot_data()
