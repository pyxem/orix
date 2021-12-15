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

from matplotlib import __version__ as _MPL_VERSION
from matplotlib import pyplot as plt
import numpy as np
import pytest

from orix.plot import RodriguesPlot, AxAnglePlot, RotationPlot
from orix.quaternion import Misorientation, Orientation, OrientationRegion
from orix.quaternion.symmetry import C1, D6


# TODO: Remove when the oldest supported version of Matplotlib
# increases from 3.3 to 3.4.
# See: https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html#mpl_toolkits.mplot3d.axes3d.Axes3D
_SUBPLOT_KWARGS = dict()
if version.parse(_MPL_VERSION) >= version.parse("3.4"):  # pragma: no cover
    _SUBPLOT_KWARGS["auto_add_to_figure"] = False


def test_init_rodrigues_plot():
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(projection="rodrigues", **_SUBPLOT_KWARGS)
    assert isinstance(ax, RodriguesPlot)


def test_init_axangle_plot():
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(projection="axangle", **_SUBPLOT_KWARGS)
    assert isinstance(ax, AxAnglePlot)


def test_RotationPlot_methods():
    """This code is lifted from demo-3-v0.1."""
    misori = Misorientation([1, 1, 1, 1])  # any will do
    ori = Orientation.random()
    fig = plt.figure()
    ax = fig.add_subplot(projection="axangle", proj_type="ortho", **_SUBPLOT_KWARGS)
    ax.scatter(misori)
    ax.scatter(ori)
    ax.plot(misori)
    ax.plot(ori)
    ax.plot_wireframe(OrientationRegion.from_symmetry(D6, D6))
    plt.close("all")

    # Clear the edge case
    ax.transform(np.asarray([1, 1, 1]))


def test_full_region_plot():
    empty = OrientationRegion.from_symmetry(C1, C1)
    _ = empty.get_plot_data()


def test_RotationPlot_transform_fundamental_zone_raises():
    fig = plt.figure()
    rp = RotationPlot(fig)
    with pytest.raises(
        TypeError, match="fundamental_zone is not an OrientationRegion object"
    ):
        rp.transform(Orientation.random(), fundamental_zone=1)


def test_RotationPlot_map_into_symmetry_reduced_zone():
    # orientations are (in, out) of D6 fundamental zone
    ori = Orientation(((1, 0, 0, 0), (0.5, 0.5, 0.5, 0.5)))
    ori.symmetry = D6
    fz = OrientationRegion.from_symmetry(ori.symmetry)
    assert np.allclose(ori < fz, (True, False))
    # test map_into_symmetry_reduced_zone in RotationPlot.transform
    fig = ori.scatter(return_figure=True)
    xyz_symmetry = fig.axes[0].collections[1]._offsets3d
    # compute same plot again but with C1 symmetry where both orientations are in C1 FZ
    ori.symmetry = C1
    fig2 = ori.scatter(return_figure=True)
    xyz = fig2.axes[0].collections[1]._offsets3d
    # test that the plotted points are not the same
    assert not np.allclose(xyz_symmetry, xyz)


def test_correct_aspect_ratio():
    # Set up figure the "old" way
    fig = plt.figure()
    ax = fig.add_subplot(projection="axangle", proj_type="ortho", **_SUBPLOT_KWARGS)

    # Check aspect ratio
    x_old, _, z_old = ax.get_box_aspect()
    assert np.allclose(x_old / z_old, 1.334, atol=1e-3)

    fr = OrientationRegion.from_symmetry(D6)
    ax._correct_aspect_ratio(fr, set_limits=False)

    x_new, _, z_new = ax.get_box_aspect()
    assert np.allclose(x_new / z_new, 3, atol=1e-3)

    # Check data limits
    assert np.allclose(ax.get_xlim(), [0, 1])
    ax._correct_aspect_ratio(fr)  # set_limits=True is default
    assert np.allclose(ax.get_xlim(), [-np.pi / 2, np.pi / 2])
