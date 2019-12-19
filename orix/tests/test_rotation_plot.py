# -*- coding: utf-8 -*-
# Copyright 2018-2019 The pyXem developers
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
from orix.plot.rotation_plot import RotationPlot,RodriguesPlot,AxAnglePlot

def test_init_RodriguesPlot():
    fig = plt.figure(figsize=(3, 3))
    _ = RodriguesPlot(fig)
    return None

def test_init_AxAnglePlot():
    fig = plt.figure(figsize=(3, 3))
    _ = AxAnglePlot(fig)
    return None

def test_RotationPlot_methods():
    #ax = RotationPlot()
    #ax.scatter()
    #ax.plot()
    #ax.plot_wireframe(OrientationRegion.from_symmetry(D6, D6))
    #ax.transform()
    return None
