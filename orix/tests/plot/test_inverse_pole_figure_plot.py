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

import matplotlib.pyplot as plt
import numpy as np

from orix.plot.inverse_pole_figure_plot import _setup_inverse_pole_figure_plot
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d


class TestInversePoleFigurePlot:
    def test_simple_setup(self):
        """Ensure that initialization of parameters is as expected."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="ipf")
        assert ax.hemisphere == "upper"  # StereographicPlot default
        assert np.allclose(ax._direction.data, (0, 0, 1))
        assert np.allclose(ax._symmetry.data, symmetry.C1.data)

        ax2 = fig.add_subplot(
            122,
            projection="ipf",
            hemisphere="lower",
            direction=Vector3d.xvector(),
            symmetry=symmetry.S6,
        )
        assert ax2.hemisphere == "lower"
        assert np.allclose(ax2._direction.data, (1, 0, 0))
        assert np.allclose(ax2._symmetry.data, symmetry.S6.data)

        assert len(fig.axes) == 2

        plt.close("all")

    def test_scatter(self):
        """Ensure that input data is handled correctly by plotting the
        same vector but passing it in all allowed data formats.
        """
        plt.rcParams["axes.grid"] = False

        point_group = symmetry.D3d
        sample_direction = Vector3d.yvector()

        fig, axes = _setup_inverse_pole_figure_plot(
            symmetry=point_group, direction=sample_direction
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert isinstance(axes[0], plt.Axes)
        ax = axes[0]

        ori = Orientation.from_euler(np.radians((325, 48, 163)))
        ori.symmetry = ax._symmetry
        # Vector
        v = ori * ax._direction
        ax.scatter(v)
        # Vector in polar coordinates as (azimuth, polar)
        polar = v.to_polar()
        polar_tuple = (polar[0].data[0], polar[1].data[0])
        ax.scatter(*polar_tuple)
        # Orientation
        ax.scatter(ori)

        # All three positions are the same
        points = ax.collections
        for i in range(len(points) - 1):
            assert np.allclose(
                points[i].get_offsets().data, points[i + 1].get_offsets().data
            )

        plt.close("all")
        plt.rcParams["axes.grid"] = True

    def test_setup_inverse_pole_figure_plot(self):
        # Default parameters
        fig1, axes1 = _setup_inverse_pole_figure_plot(symmetry=symmetry.C6h)
        assert axes1.size == 1
        ax1 = axes1[0]
        assert np.allclose(ax1._direction.data, (0, 0, 1))
        assert ax1.hemisphere == "upper"

        # Ask for both hemispheres
        fig2, axes2 = _setup_inverse_pole_figure_plot(
            symmetry=symmetry.T, hemisphere="both"
        )
        assert axes2.size == 2
        hemispheres = ["upper", "lower"]
        for a, hemi in zip(axes2, hemispheres):
            assert a.hemisphere == hemi

        fig3, axes3 = _setup_inverse_pole_figure_plot(
            symmetry=symmetry.Oh, hemisphere="lower"
        )
        assert axes3[0].hemisphere == "lower"

        # Grid of two rows, three columns, one column per direction with
        # both hemispheres
        directions = Vector3d(((1, 0, 0), (0, 1, 0), (0, 1, 1)))
        fig4, axes4 = _setup_inverse_pole_figure_plot(
            symmetry=symmetry.C4, hemisphere="both", direction=directions
        )
        assert axes4.size == 6
        assert axes4[0].get_subplotspec().get_gridspec().get_geometry() == (2, 3)  # Puh
        hemispheres = ["upper"] * 3 + ["lower"] * 3
        titles = ["x", "y", "0 1 1"] * 2
        for a, hemi, title in zip(axes4, hemispheres, titles):
            assert a.hemisphere == hemi
            assert a.title.get_text() == title

        plt.close("all")
