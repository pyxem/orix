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
    StereographicTransform,
    TwoFoldMarker,
    ThreeFoldMarker,
    FourFoldMarker,
    SixFoldMarker,
)
from orix import plot, vector


plt.rcParams["backend"] = "TkAgg"
plt.rcParams["axes.grid"] = True
PROJ_NAME = "stereographic"


class TestStereographicPlot:
    def test_scatter(self):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        assert ax.name == PROJ_NAME
        assert ax._polar_cap == 0.5 * np.pi
        assert ax._azimuth_cap == 2 * np.pi
        assert ax._polar_resolution == 10
        assert ax._azimuth_resolution == 10
        assert ax.get_data_ratio() == 1
        assert ax.can_pan() is False
        assert ax.can_zoom() is False

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

    def test_transform_path_non_affine(self):
        # This is just to get this part covered
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        spine_path = ax.spines["stereographic"]._path
        st = StereographicTransform()
        spine_path_transformed = st.transform_path_non_affine(spine_path)
        assert np.allclose(spine_path_transformed.vertices[0], [-0.5463, 0], atol=1e-4)

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

        ax.azimuth_grid()
        ax.polar_grid()
        assert ax._azimuth_resolution == azimuth_res
        assert ax._polar_resolution == polar_res

        ax.azimuth_grid(30)
        ax.polar_grid(45)
        assert ax._azimuth_resolution == 30
        assert ax._polar_resolution == 45

        plt.close("all")

    def test_set_labels(self):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        assert ax.texts == []

        ax.set_labels(None, None, None)
        assert ax.texts == []

        ax.set_labels("X", None, None)
        assert len(ax.texts) == 1
        assert ax.texts[0].get_text() == "X"
        assert np.allclose([ax.texts[0]._x, ax.texts[0]._y], [0, 0.5 * np.pi])

        ax.set_labels(None, "TD", None)
        assert len(ax.texts) == 2
        assert ax.texts[1].get_text() == "TD"
        assert np.allclose([ax.texts[1]._x, ax.texts[1]._y], [0.5 * np.pi, 0.5 * np.pi])

        ax.hemisphere = "lower"
        ax.set_labels(False, False, color="xkcd:salmon")
        assert len(ax.texts) == 3
        assert ax.texts[2].get_text() == "z"
        assert ax.texts[2].get_color() == "xkcd:salmon"
        assert np.allclose([ax.texts[2]._x, ax.texts[2]._y], [0, np.pi])

        plt.close("all")

    def test_show_hemisphere_label(self):
        _, ax = plt.subplots(ncols=2, subplot_kw=dict(projection=PROJ_NAME))

        ax[0].scatter(vector.Vector3d([0, 0, 1]))
        ax[0].show_hemisphere_label()
        label_up = ax[0].texts[0]
        assert label_up.get_text() == "upper"
        assert label_up.get_color() == "black"
        assert np.allclose([label_up._x, label_up._y], [2.356, 1.571], atol=1e-3)

        ax[1].hemisphere = "lower"
        ax[1].scatter(vector.Vector3d([0, 0, -1]))
        ax[1].show_hemisphere_label(color="r")
        label_low = ax[1].texts[0]
        assert label_low.get_text() == "lower"
        assert label_low.get_color() == "r"
        assert np.allclose([label_low._x, label_low._y], [2.356, 1.571], atol=1e-3)

        plt.close("all")

    @pytest.mark.parametrize(
        "hemisphere, pole, hemi_str", [("uPPer", -1, "upper"), ("loweR", 1, "lower")],
    )
    def test_hemisphere_pole(self, hemisphere, pole, hemi_str):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        assert ax.hemisphere == "upper"
        assert ax.pole == ax.transProjection.pole == ax.transAffine.pole == -1

        ax.hemisphere = hemisphere
        assert ax.hemisphere == hemi_str
        assert ax.pole == ax.transProjection.pole == ax.transAffine.pole == pole

        plt.close("all")

    def test_hemisphere_raises(self):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        with pytest.raises(ValueError, match="Hemisphere must be 'upper' or"):
            ax.hemisphere = "west"

        plt.close("all")

    def test_format_coord(self):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        for a, p in [(0, 0), (0.75 * np.pi, 0.25 * np.pi)]:
            assert ax.format_coord(a, p) == (
                "\N{GREEK SMALL LETTER PHI}={:.2f}\N{GREEK SMALL LETTER PI} "
                "({:.2f}\N{DEGREE SIGN}), "
                "\N{GREEK SMALL LETTER theta}={:.2f}\N{GREEK SMALL LETTER PI} "
                "({:.2f}\N{DEGREE SIGN})"
            ).format(a / np.pi, np.rad2deg(a), p / np.pi, np.rad2deg(p))

        plt.close("all")

    def test_empty_scatter(self):
        v = vector.Vector3d([0, 0, 1])

        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        ax.hemisphere = "lower"

        # Not plotted since the vector isn't visible in this hemisphere
        ax.scatter(v)
        ax.text(v, s="1")
        assert ax.texts == []

        plt.close("all")

    @pytest.mark.parametrize("shape", [(5, 10), (2, 3)])
    def test_multidimensional_vector(self, shape):
        n = np.prod(shape)
        v = vector.Vector3d(np.random.normal(size=3 * n).reshape(shape + (3,)))
        v.scatter()
        v.draw_circle()

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


class TestStereographicTransform:
    @pytest.mark.parametrize(
        "pole, azimuth_polar, xy",
        [
            (
                -1,
                [[0, 0], [0.5 * np.pi, 0.5 * np.pi], [0.25 * np.pi, 0.5 * np.pi]],
                [[0, 0], [0, 1], [0.5 * np.sqrt(2), 0.5 * np.sqrt(2)]],
            ),
            (
                1,
                [[0, np.pi], [0.5 * np.pi, 0.5 * np.pi], [0.25 * np.pi, 0.75 * np.pi]],
                [[0, 0], [0, 1], [1 / (np.sqrt(2) + 2), 1 / (np.sqrt(2) + 2)]],
            ),
        ],
    )
    def test_transform(self, pole, azimuth_polar, xy):
        st = StereographicTransform(pole=pole)
        sti = st.inverted()
        assert st.pole == sti.pole == pole
        for ap, xyi in zip(azimuth_polar, xy):
            assert np.allclose(st.transform(ap), xyi)
            assert np.allclose(sti.transform(xyi), ap)

    def test_transform_inverted_loop(self):
        st = StereographicTransform()
        assert np.allclose(
            st.inverted().inverted().transform((0.25 * np.pi, 0.5 * np.pi)),
            (0.5 * np.sqrt(2), 0.5 * np.sqrt(2)),
        )


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
        "hemisphere, polar_cap, polar, desired_array",
        [
            (
                "upper",
                0.5 * np.pi,
                np.deg2rad([60, 90, 120, 150, 180, 30]),
                np.array([1, 1, 0, 0, 0, 1], dtype=bool),
            ),
            (
                "lower",
                0.5 * np.pi,
                np.deg2rad([60, 90, 120, 150, 180, 30]),
                np.array([0, 0, 1, 1, 1, 0], dtype=bool),
            ),
        ],
    )
    def test_visible_in_hemisphere(self, hemisphere, polar_cap, polar, desired_array):
        assert np.allclose(
            plot.stereographic_plot._visible_in_hemisphere(
                hemisphere=hemisphere, polar_cap=polar_cap, polar=polar
            ),
            desired_array,
        )

    @pytest.mark.parametrize(
        "hemisphere, polar_cap, azimuth, polar, desired_azimuth, desired_polar",
        [
            (
                "upper",
                0.5 * np.pi,
                np.arange(5),
                np.deg2rad([60, 90, 120, 150, 180, 30]),
                np.roll(np.arange(5), shift=-5),
                np.roll(np.deg2rad([60, 90, 120, 150, 180, 30]), shift=-5),
            ),
            (
                "lower",
                0.5 * np.pi,
                np.arange(5),
                np.deg2rad([60, 90, 120, 150, 180, 30]),
                np.roll(np.arange(5), shift=-5),
                np.roll(np.deg2rad([60, 90, 120, 150, 180, 30]), shift=-5),
            ),
        ],
    )
    def test_sort_coords_by_shifted_bools(
        self, hemisphere, polar_cap, azimuth, polar, desired_azimuth, desired_polar
    ):
        azimuth_out, polar_out = plot.stereographic_plot._sort_coords_by_shifted_bools(
            hemisphere=hemisphere, polar_cap=polar_cap, azimuth=azimuth, polar=polar,
        )
        assert np.allclose(azimuth_out, desired_azimuth)
        assert np.allclose(polar_out, desired_polar)

    def test_draw_circle(self):
        v1 = vector.Vector3d([[0, 0, 1], [1, 0, 1], [1, 1, 1]])
        v2 = vector.Vector3d(np.append(v1.data, -v1.data, axis=0))

        _, ax = plt.subplots(ncols=2, subplot_kw=dict(projection=PROJ_NAME))
        c = [f"C{i}" for i in range(6)]
        ax[0].scatter(v2, c=c)
        ax[0].draw_circle(v2, color=c, steps=100)
        ax[1].hemisphere = "lower"
        ax[1].scatter(v2, c=c)
        ax[1].draw_circle(v2, color=c, steps=150, linewidth=3)

        # Circles
        assert len(ax[0].lines) == len(ax[1].lines) == 3
        assert ax[0].lines[0]._path._vertices.shape == (100, 2)
        assert ax[1].lines[0]._path._vertices.shape == (150, 2)

    def test_draw_circle_empty(self):
        v1 = vector.Vector3d([[0, 0, 1], [1, 0, 1], [1, 1, 1]])
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        ax.hemisphere = "lower"
        ax.draw_circle(v1)
        assert len(ax.lines) == 0
