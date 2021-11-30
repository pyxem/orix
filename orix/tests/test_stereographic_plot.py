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

import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest

from orix.plot.stereographic_plot import (
    TwoFoldMarker,
    ThreeFoldMarker,
    FourFoldMarker,
    SixFoldMarker,
)
from orix import plot, vector
from orix.quaternion.symmetry import C1, C6, Oh


plt.rcParams["backend"] = "TkAgg"
plt.rcParams["axes.grid"] = True
PROJ_NAME = "stereographic"


class TestStereographicPlot:
    def test_scatter(self):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        assert ax.name == PROJ_NAME
        assert ax._polar_resolution == 10
        assert ax._azimuth_resolution == 10
        assert ax.get_data_ratio() == 1
        assert ax.can_pan()
        assert ax.can_zoom()

        v = vector.Vector3d([[0, 0, 1], [2, 0, 2]])
        ax.scatter(v[0])
        ax.scatter(v[1].azimuth.data, v[1].polar.data)

        with pytest.raises(ValueError, match="Accepts only one "):
            ax.scatter(v[0].azimuth.data)

        plt.close("all")

    def test_annotate(self):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        v = vector.Vector3d([[0, 0, 1], [-1, 0, 1], [1, 1, 1]])
        ax.scatter(v)
        format_vector = lambda v: str(v.data[0]).replace(" ", "")
        for vi in v:
            ax.text(vi, s=format_vector(vi))

        assert len(ax.texts) == 3
        assert ax.texts[0].get_text() == "[001]"
        assert ax.texts[1].get_text() == "[-101]"
        assert ax.texts[2].get_text() == "[111]"

        plt.close("all")

    def test_great_circle_equator(self):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        n = 100
        azimuth = np.linspace(0, 2 * np.pi, n)
        polar = np.ones(n) * 0.5 * np.pi
        ax.plot(azimuth, polar)

        plt.close("all")

    def test_grids(self):
        azimuth_res = 10
        polar_res = 15
        _, ax = plt.subplots(
            subplot_kw=dict(
                projection=PROJ_NAME,
                azimuth_resolution=azimuth_res,
                polar_resolution=polar_res,
            )
        )
        assert ax._azimuth_resolution == azimuth_res
        assert ax._polar_resolution == polar_res

        ax.stereographic_grid()
        assert ax._azimuth_resolution == azimuth_res
        assert ax._polar_resolution == polar_res

        ax.stereographic_grid(azimuth_resolution=30, polar_resolution=45)
        assert ax._azimuth_resolution == 30
        assert ax._polar_resolution == 45

        plt.close("all")

    def test_set_labels(self):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        assert len(ax.texts) == 0

        ax.set_labels(None, None, None)
        assert len(ax.texts) == 0

        ax.set_labels("X", None, None)
        assert len(ax.texts) == 1
        assert ax.texts[0].get_text() == "X"
        assert np.allclose([ax.texts[0]._x, ax.texts[0]._y], [1, 0])

        ax.set_labels(None, "TD", None)
        assert len(ax.texts) == 2
        assert ax.texts[1].get_text() == "TD"
        assert np.allclose([ax.texts[1]._x, ax.texts[1]._y], [0, 1])

        ax.hemisphere = "lower"
        ax.set_labels(False, False, color="xkcd:salmon")
        assert len(ax.texts) == 3
        assert ax.texts[2].get_text() == "z"
        assert ax.texts[2].get_color() == "xkcd:salmon"
        assert np.allclose([ax.texts[2]._x, ax.texts[2]._y], [0, 0])

        plt.close("all")

    def test_show_hemisphere_label(self):
        _, ax = plt.subplots(ncols=2, subplot_kw=dict(projection=PROJ_NAME))

        label_xy = [-0.71, 0.71]

        ax[0].scatter(vector.Vector3d([0, 0, 1]))
        ax[0].show_hemisphere_label()
        label_up = ax[0].texts[0]
        assert label_up.get_text() == "upper"
        assert label_up.get_color() == "black"
        assert np.allclose([label_up._x, label_up._y], label_xy)

        ax[1].hemisphere = "lower"
        ax[1].scatter(vector.Vector3d([0, 0, -1]))
        ax[1].show_hemisphere_label(color="r")
        label_low = ax[1].texts[0]
        assert label_low.get_text() == "lower"
        assert label_low.get_color() == "r"
        assert np.allclose([label_low._x, label_low._y], label_xy)

        plt.close("all")

    @pytest.mark.parametrize(
        "hemisphere, pole, hemi_str", [("uPPer", -1, "upper"), ("loweR", 1, "lower")]
    )
    def test_hemisphere_pole(self, hemisphere, pole, hemi_str):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        assert ax.hemisphere == "upper"
        assert ax.pole == ax._projection.pole == -1

        ax.hemisphere = hemisphere
        assert ax.hemisphere == hemi_str
        assert ax.pole == ax._projection.pole == pole

        plt.close("all")

    def test_hemisphere_raises(self):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        with pytest.raises(ValueError, match="Hemisphere must be 'upper' or"):
            ax.hemisphere = "west"

        plt.close("all")

    def test_format_coord(self):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        xy = [(0, 0), (-0.2929, 0.2929)]
        spherical = [(0, 0), (0.75 * np.pi, 0.25 * np.pi)]
        for (x, y), (a, p) in zip(xy, spherical):
            assert ax.format_coord(x, y) == (
                "\N{GREEK SMALL LETTER PHI}={:.2f}\N{GREEK SMALL LETTER PI} "
                "({:.2f}\N{DEGREE SIGN}), "
                "\N{GREEK SMALL LETTER theta}={:.2f}\N{GREEK SMALL LETTER PI} "
                "({:.2f}\N{DEGREE SIGN})"
            ).format(a / np.pi, np.rad2deg(a), p / np.pi, np.rad2deg(p))

        assert ax.format_coord(1, 1) == ""

        plt.close("all")

    def test_empty_scatter(self):
        v = vector.Vector3d([0, 0, 1])

        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        ax.hemisphere = "lower"

        # Not plotted since the vector isn't visible in this hemisphere
        ax.scatter(v)
        ax.text(v, s="1")
        assert len(ax.texts) == 0

        plt.close("all")

    @pytest.mark.parametrize("shape", [(5, 10), (2, 3)])
    def test_multidimensional_vector(self, shape):
        n = np.prod(shape)
        v = vector.Vector3d(np.random.normal(size=3 * n).reshape(shape + (3,)))
        v.scatter()
        v.draw_circle()

        plt.close("all")

    def test_order_in_hemisphere(self):
        v = vector.Vector3d.from_polar(
            azimuth=np.radians([45, 90, 135, 180]),
            polar=np.radians([50, 45, 140, 135]),
        )

        fig, ax = plt.subplots(ncols=2, subplot_kw=dict(projection=PROJ_NAME))
        ax[1].hemisphere = "lower"
        x_upper, y_upper, visible_upper = ax[0]._pretransform_input((v,), sort=True)
        x_lower, y_lower, visible_lower = ax[1]._pretransform_input((v,), sort=True)

        x_upper_desired, y_upper_desired = ax[0]._projection.vector2xy(v[:2])
        assert np.allclose(x_upper, x_upper_desired)
        assert np.allclose(y_upper, y_upper_desired)

        x_lower_desired, y_lower_desired = ax[1]._projection.vector2xy(v[2:])
        assert np.allclose(x_lower, x_lower_desired)
        assert np.allclose(y_lower, y_lower_desired)

        assert np.allclose(visible_upper, [True, True, False, False])
        assert np.allclose(visible_lower, [False, False, True, True])

        x_upper2, y_upper2, visible_upper2 = ax[0]._pretransform_input(
            (v[2:],), sort=True
        )
        assert x_upper2.size == y_upper2.size == 0
        assert not visible_upper2.any()

        plt.close("all")


class TestSymmetryMarker:
    def test_properties(self):
        v2fold = vector.Vector3d([[1, 0, 1], [0, 1, 1]])
        marker2fold = TwoFoldMarker(v2fold)
        assert np.allclose(v2fold.data, marker2fold._vector.data)
        assert marker2fold.fold == 2
        assert marker2fold.n == 2
        assert np.allclose(marker2fold.size, [1.55, 1.55], atol=1e-2)
        assert isinstance(marker2fold._marker[0], mpath.Path)

        v3fold = vector.Vector3d([1, 1, 1])
        marker3fold = ThreeFoldMarker(v3fold, size=5)
        assert np.allclose(v3fold.data, marker3fold._vector.data)
        assert marker3fold.fold == 3
        assert marker3fold.n == 1
        assert np.allclose(marker3fold.size, 5)

        # Iterating over markers
        for i, (vec, mark, size) in enumerate(marker3fold):
            assert np.allclose(vec.data, v3fold[i].data)
            assert np.allclose(mark, (3, 0, 45 + 90))
            assert size == 5

        v4fold = vector.Vector3d([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        marker4fold = FourFoldMarker(v4fold, size=11)
        assert np.allclose(v4fold.data, marker4fold._vector.data)
        assert marker4fold.fold == 4
        assert marker4fold.n == 3
        assert np.allclose(marker4fold.size, [11, 11, 11])
        assert marker4fold._marker == ["D"] * 3

        marker6fold = SixFoldMarker([0, 0, 1], size=15)
        assert isinstance(marker6fold._vector, vector.Vector3d)
        assert np.allclose(marker6fold._vector.data, [0, 0, 1])
        assert marker6fold.fold == 6
        assert marker6fold.n == 1
        assert marker6fold.size == 15
        assert marker6fold._marker == ["h"]

        plt.close("all")

    def test_plot_symmetry_marker(self):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        ax.stereographic_grid(False)
        marker_size = 500

        v4fold = vector.Vector3d(
            [[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        )
        ax.symmetry_marker(v4fold, fold=4, c="C4", s=marker_size)

        v3fold = vector.Vector3d([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]])
        ax.symmetry_marker(v3fold, fold=3, c="C3", s=marker_size)

        v2fold = vector.Vector3d(
            [
                [1, 0, 1],
                [0, 1, 1],
                [-1, 0, 1],
                [0, -1, 1],
                [1, 1, 0],
                [-1, -1, 0],
                [-1, 1, 0],
                [1, -1, 0],
            ]
        )
        ax.symmetry_marker(v2fold, fold=2, c="C2", s=marker_size)

        ax.symmetry_marker([0, 0, 1], fold=6, s=marker_size)

        markers = ax.collections
        assert len(markers) == 18
        assert np.allclose(markers[0]._sizes, marker_size)
        assert np.allclose(markers[-1]._sizes, marker_size)
        assert np.allclose(markers[0]._facecolors, mcolors.to_rgba("C4"))
        assert np.allclose(markers[5]._facecolors, mcolors.to_rgba("C3"))
        assert np.allclose(markers[-2]._facecolors, mcolors.to_rgba("C2"))
        assert np.allclose(markers[-1]._facecolors, mcolors.to_rgba("C0"))

        with pytest.raises(ValueError, match="Can only plot 2"):
            ax.symmetry_marker([0, 0, 1], fold=5)

        plt.close("all")


class TestDrawCircle:
    @pytest.mark.parametrize(
        "value, visible, desired_array",
        [
            ("C0", np.array([1, 1, 1, 0, 1], dtype=bool), ["C0"] * 4),
            (["C0", "C1"], np.array([1, 1, 1, 0, 1], dtype=bool), ["C0"] * 4),
        ],
    )
    def test_get_array_of_values(self, value, visible, desired_array):
        assert all(
            plot.stereographic_plot._get_array_of_values(value=value, visible=visible)
            == desired_array
        )

    @pytest.mark.parametrize(
        "pole, polar, desired_array",
        [
            (
                -1,
                np.deg2rad([60, 90, 120, 150, 180, 30]),
                np.array([1, 1, 0, 0, 0, 1], dtype=bool),
            ),
            (
                1,
                np.deg2rad([60, 90, 120, 150, 180, 30]),
                np.array([0, 1, 1, 1, 1, 0], dtype=bool),
            ),
        ],
    )
    def test_visible_in_hemisphere(self, pole, polar, desired_array):
        assert np.allclose(
            plot.stereographic_plot._is_visible(polar, pole), desired_array
        )

    def test_draw_circle(self):
        v1 = vector.Vector3d([[0, 0, 1], [1, 0, 1], [1, 1, 1]])
        v2 = vector.Vector3d(np.append(v1.data, -v1.data, axis=0))

        upper_steps = 100
        lower_steps = 150

        _, ax = plt.subplots(ncols=2, subplot_kw=dict(projection=PROJ_NAME))
        c = [f"C{i}" for i in range(6)]
        ax[0].scatter(v2, c=c)
        ax[0].draw_circle(v2, color=c, steps=upper_steps)
        ax[1].hemisphere = "lower"
        ax[1].scatter(v2, c=c)
        ax[1].draw_circle(v2, color=c, steps=lower_steps, linewidth=3)

        # Circles
        assert len(ax[0].lines) == 3
        assert len(ax[1].lines) == 3
        assert ax[0].lines[0]._path._vertices.shape == (upper_steps, 2)
        assert ax[1].lines[0]._path._vertices.shape == (lower_steps, 2)
        assert ax[1].lines[1]._path._vertices.shape == (lower_steps // 2 + 1, 2)
        assert ax[1].lines[1]._path._vertices.shape == (lower_steps // 2 + 1, 2)

    def test_draw_circle_empty(self):
        v1 = vector.Vector3d([[0, 0, 1], [1, 0, 1], [1, 1, 1]])
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        ax.hemisphere = "lower"
        ax.draw_circle(v1)
        assert len(ax.lines) == 0


class TestRestrictToFundamentalSector:
    def test_restrict_to_fundamental_sector(self):
        fig1, ax1 = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        vertices = ax1.patches[0].get_verts()

        # C1's has no fundamental sector, so the circle marking the
        # edge of the axis region should be unchanged
        fig2, ax2 = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        ax2.restrict_to_sector(C1.fundamental_sector)
        assert np.allclose(vertices, ax2.patches[0].get_verts())

        # C6's fundamental sector is 1 / 6 of the unit sphere, with
        # half of it in the upper hemisphere
        fig3, ax3 = plt.subplots(ncols=2, subplot_kw=dict(projection=PROJ_NAME))
        ax3[0].restrict_to_sector(C6.fundamental_sector)
        assert not np.allclose(vertices, ax3[0].patches[0].get_verts())
        assert ax3[0].patches[1].get_label() == "sa_sector"

        # Ensure grid lines are clipped by sector
        ax3[0].stereographic_grid(False)
        ax3[0].stereographic_grid(True)

        # Oh's fundamental sector is only in the upper hemisphere,
        # so the same as C1's sector applies for the lower hemisphere
        fs = Oh.fundamental_sector
        fig4, ax4 = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        ax4.restrict_to_sector(fs)
        upper_patches4 = ax4.patches
        assert len(upper_patches4) == 2
        assert upper_patches4[1].get_label() == "sa_sector"

        fig5, ax5 = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        ax5.hemisphere = "lower"
        ax5.restrict_to_sector(fs)
        lower_patches4 = ax5.patches
        assert len(lower_patches4) == 1
        assert lower_patches4[0].get_label() == "sa_circle"
        assert np.allclose(vertices, lower_patches4[0].get_verts())

        # No lines are added to lower hemisphere, only the upper
        assert len(ax4.lines) == 0
        assert len(ax5.lines) == 0
        ax4.plot(fs.edges)
        ax5.plot(fs.edges)
        assert len(ax4.lines) == 1
        assert len(ax5.lines) == 0

        plt.close("all")
