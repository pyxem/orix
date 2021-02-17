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
import pytest

from orix.plot.stereographic_plot import StereographicTransform
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
        assert ax._polar_resolution == 30
        assert ax._azimuth_resolution == 30
        assert ax.get_data_ratio() == 1
        assert ax.can_pan() is False
        assert ax.can_zoom() is False

        v = vector.Vector3d([[0, 0, 1], [2, 0, 2]])
        ax.scatter(v[0])
        ax.scatter(v[1].phi.data, v[1].theta.data)

        with pytest.raises(ValueError, match="Accepts only one "):
            ax.scatter(v[1].phi)

        plt.close("all")

    def test_annotate(self):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        v = vector.Vector3d([[0, 0, 1], [-1, 0, 1], [1, 1, 1]])
        ax.scatter(v)
        v_str = v[:2]._nice_string_repr()
        for vi, vi_str in zip(v[:2], v_str):
            ax.text(vi, s=vi_str)
        ax.text(v[2], s=v[2]._nice_string_repr("()"))

        assert len(ax.texts) == 3
        assert ax.texts[0]._text == "[001]"
        assert ax.texts[1]._text == "[-101]"
        assert ax.texts[2]._text == "(111)"

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
        assert ax.texts[0]._text == "X"
        assert np.allclose([ax.texts[0]._x, ax.texts[0]._y], [0, 0.5 * np.pi])

        ax.set_labels(None, "TD", None)
        assert len(ax.texts) == 2
        assert ax.texts[1]._text == "TD"
        assert np.allclose([ax.texts[1]._x, ax.texts[1]._y], [0.5 * np.pi, 0.5 * np.pi])

        ax.hemisphere = "lower"
        ax.set_labels(False, False, color="xkcd:salmon")
        assert len(ax.texts) == 3
        assert ax.texts[2]._text == "Z"
        assert ax.texts[2]._color == "xkcd:salmon"
        assert np.allclose([ax.texts[2]._x, ax.texts[2]._y], [0, np.pi])

        plt.close("all")

    def test_show_hemisphere_label(self):
        _, ax = plt.subplots(ncols=2, subplot_kw=dict(projection=PROJ_NAME))

        ax[0].scatter(vector.Vector3d([0, 0, 1]))
        ax[0].show_hemisphere_label()
        label_up = ax[0].texts[0]
        assert label_up._text == "upper"
        assert label_up._color == "black"
        assert np.allclose([label_up._x, label_up._y], [2.356, 1.571], atol=1e-3)

        ax[1].hemisphere = "south"
        ax[1].scatter(vector.Vector3d([0, 0, -1]))
        ax[1].show_hemisphere_label(color="r")
        label_low = ax[1].texts[0]
        assert label_low._text == "lower"
        assert label_low._color == "r"
        assert np.allclose([label_low._x, label_low._y], [2.356, 1.571], atol=1e-3)

        plt.close("all")

    @pytest.mark.parametrize(
        "hemisphere, pole, hemi_str",
        [
            ("upper", -1, "upper"),
            ("north", -1, "upper"),
            ("lower", 1, "lower"),
            ("south", 1, "lower"),
        ],
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
        with pytest.raises(ValueError, match="Hemisphere must be upper/north or"):
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


class TestSymmetryMarker:
    def test_point_group_432(self):
        _, ax = plt.subplots(subplot_kw=dict(projection=PROJ_NAME))
        marker_size = 500
        v4fold = vector.Vector3d(
            [[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        )
        ax.symmetry_marker(v4fold, fold=4, c="C4", s=marker_size)


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
