# -*- coding: utf-8 -*-
# Copyright 2018-2023 the orix developers
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

from matplotlib import __version__ as mpl_version
from matplotlib import pyplot as plt
import numpy as np
from packaging import version
import pytest

from orix.plot import AxAnglePlot, RodriguesPlot
from orix.quaternion import Misorientation, Orientation, OrientationRegion
from orix.quaternion.symmetry import C1, D6, _proper_groups
from orix.vector import AxAngle, Rodrigues

# TODO: Remove when the oldest supported version of Matplotlib
#  increases from 3.3 to 3.4.
# See: https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html#mpl_toolkits.mplot3d.axes3d.Axes3D
subplot_kw = dict()
if version.parse(mpl_version) >= version.parse("3.4"):  # pragma: no cover
    subplot_kw["auto_add_to_figure"] = False


class TestRotationPlot:
    def test_RotationPlot_methods(self):
        """This code is lifted from demo-3-v0.1."""
        misori = Misorientation([1, 1, 1, 1])  # any will do
        ori = Orientation.random()
        fig = plt.figure()
        ax = fig.add_subplot(projection="axangle", proj_type="ortho", **subplot_kw)
        ax.scatter(misori)
        ax.scatter(ori)
        ax.plot(misori)
        ax.plot(ori)
        ax.plot_wireframe(OrientationRegion.from_symmetry(D6, D6))
        plt.close("all")

        # Clear the edge case
        ax.transform(np.asarray([1, 1, 1]))

        plt.close("all")

    def test_full_region_plot(self):
        empty = OrientationRegion.from_symmetry(C1, C1)
        _ = empty.get_plot_data()

    def test_transform_fundamental_zone_raises(self):
        fig = plt.figure()
        rp = fig.add_subplot(projection="axangle")
        with pytest.raises(
            TypeError, match="fundamental_zone is not an OrientationRegion"
        ):
            _ = rp.transform(Orientation.random(), fundamental_zone=1)

        plt.close("all")

    def test_reduce(self):
        # Orientations are (in, out) of D6 fundamental zone
        ori = Orientation(((1, 0, 0, 0), (0.5, 0.5, 0.5, 0.5)), symmetry=D6)
        fz = OrientationRegion.from_symmetry(ori.symmetry)
        assert np.allclose(ori < fz, (True, False))

        # Test reduce() in RotationPlot.transform
        fig = ori.scatter(return_figure=True)
        xyz_symmetry = fig.axes[0].collections[1]._offsets3d

        # Compute same plot again but with C1 symmetry where both
        # orientations are in C1 FZ
        ori.symmetry = C1
        fig2 = ori.scatter(return_figure=True)
        xyz = fig2.axes[0].collections[1]._offsets3d
        assert not np.allclose(xyz_symmetry, xyz)

        plt.close("all")

    def test_correct_aspect_ratio(self):
        # Set up figure the "old" way
        fig = plt.figure()
        ax = fig.add_subplot(projection="axangle", proj_type="ortho", **subplot_kw)

        # Check aspect ratio
        x_old, _, z_old = ax.get_box_aspect()
        assert np.allclose(x_old / z_old, 1.334, atol=1e-3)

        fz = OrientationRegion.from_symmetry(D6)
        ax._correct_aspect_ratio(fz, set_limits=False)

        x_new, _, z_new = ax.get_box_aspect()
        assert np.allclose(x_new / z_new, 3, atol=1e-3)

        # Check data limits
        assert np.allclose(ax.get_xlim(), [0, 1])
        ax._correct_aspect_ratio(fz)  # set_limits=True is default
        assert np.allclose(ax.get_xlim(), [-np.pi / 2, np.pi / 2])

        plt.close("all")


class TestRodriguesPlot:
    def test_initialize_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="rodrigues", **subplot_kw)
        assert isinstance(ax, RodriguesPlot)

        plt.close("all")

    @pytest.mark.parametrize("sym", _proper_groups)
    def test_vector_coordinates(self, sym):
        """Coordinate transformation method returns Rodrigues vectors
        with expected coordinates.
        """
        o = Orientation.random((30,))
        o.symmetry = sym
        rod = Rodrigues.from_rotation(o.reduce())

        fig = o.scatter("rodrigues", return_figure=True)
        ax = fig.axes[0]
        xyz = ax.transform(o)
        rod2 = np.stack(xyz, axis=1)
        assert np.allclose(rod.data, rod2.data)

        xyz = ax.collections[1]._offsets3d
        rod3 = np.stack(xyz, axis=1)
        assert np.allclose(rod.data, rod3.data)

        plt.close("all")


class TestAxAnglePlot:
    def test_initialize_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="axangle", **subplot_kw)
        assert isinstance(ax, AxAnglePlot)

        plt.close("all")

    @pytest.mark.parametrize("sym", _proper_groups)
    def test_vector_coordinates(self, sym):
        """Coordinate transformation method returns axis-angle vectors
        with expected coordinates.
        """
        o = Orientation.random((30,))
        o.symmetry = sym
        rod = AxAngle.from_rotation(o.reduce())

        fig = o.scatter(return_figure=True)
        ax = fig.axes[0]
        xyz = ax.transform(o)
        rod2 = np.stack(xyz, axis=1)
        assert np.allclose(rod.data, rod2.data)

        xyz = ax.collections[1]._offsets3d
        rod3 = np.stack(xyz, axis=1)
        assert np.allclose(rod.data, rod3.data)

        plt.close("all")
