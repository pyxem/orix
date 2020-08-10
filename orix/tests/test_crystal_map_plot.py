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

import copy

import numpy as np
import matplotlib.pyplot as plt
import pytest

from orix.plot import CrystalMapPlot
from orix.plot.crystal_map_plot import convert_unit
from orix.crystal_map import CrystalMap
from orix.crystal_map.phase_list import PhaseList

plt.rcParams["backend"] = "TkAgg"

# Warning raised when plot width is too small (this too is tested)
SCALEBAR_WARNING = "Scalebar width .* is not an integer."

# Can be easily changed in the future
PLOT_MAP = "plot_map"


class TestCrystalMapPlot:
    @pytest.mark.parametrize(
        "crystal_map_input, expected_data_shape",
        [
            (((1, 10, 20), (0, 1.5, 1.5), 1, [0]), (10, 20, 3)),
            (((1, 4, 3), (0, 0.1, 0.1), 1, [0]), (4, 3, 3)),
        ],
        indirect=["crystal_map_input"],
    )
    @pytest.mark.filterwarnings(f"ignore: {SCALEBAR_WARNING}")
    def test_plot_phase(self, crystal_map_input, phase_list, expected_data_shape):
        cm = CrystalMap(**crystal_map_input)

        assert np.unique(cm.phase_id) == np.array([0])  # Test code assumption
        assert phase_list.ids == [0, 1, 2]
        cm.phases = phase_list
        cm[0, 0].phase_id = 0
        cm[1, 1].phase_id = 2

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        im = ax.plot_map(cm)

        # Expected image data
        phase_id = cm.get_map_data("phase_id")
        unique_phase_ids = np.unique(phase_id[~np.isnan(phase_id)])
        expected_data = np.ones(phase_id.shape + (3,))
        for i, color in zip(unique_phase_ids, cm.phases_in_data.colors_rgb):
            mask = phase_id == int(i)
            expected_data[mask] = expected_data[mask] * color

        image_data = im.get_array()
        assert np.allclose(image_data.shape, expected_data_shape)
        assert np.allclose(image_data, expected_data)

        plt.close("all")

    @pytest.mark.filterwarnings(f"ignore: {SCALEBAR_WARNING}")
    def test_plot_property(self, crystal_map):
        cm = crystal_map

        prop_name = "iq"
        prop_data = np.arange(cm.size)
        cm.prop[prop_name] = prop_data

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        im = ax.plot_map(cm, cm.iq)

        assert np.allclose(im.get_array(), prop_data.reshape(cm.shape))

        plt.close("all")

    @pytest.mark.filterwarnings(f"ignore: {SCALEBAR_WARNING}")
    def test_plot_scalar(self, crystal_map):
        cm = crystal_map

        angles = cm.rotations.angle.data

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection=PLOT_MAP)
        # Test use of `vmax` as well
        im1 = ax1.plot_map(cm, angles, vmax=angles.max() - 10)

        assert np.allclose(im1.get_array(), angles.reshape(cm.shape), atol=1e-3)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(projection=PLOT_MAP)
        im2 = ax2.plot_map(cm, cm.rotations.angle)

        assert np.allclose(im2.get_array(), angles.reshape(cm.shape), atol=1e-3)

        plt.close("all")

    @pytest.mark.parametrize(
        "crystal_map_input",
        [((2, 9, 3), (1, 1.5, 1.5), 1, [0]), ((2, 10, 5), (1, 0.1, 0.1), 1, [0])],
        indirect=["crystal_map_input"],
    )
    @pytest.mark.filterwarnings(f"ignore: {SCALEBAR_WARNING}")
    def test_plot_masked_phase(self, crystal_map_input, phase_list):
        cm = CrystalMap(**crystal_map_input)

        # Test code assumptions
        assert np.unique(cm.phase_id) == np.array([0])
        assert phase_list.ids == [0, 1, 2]
        assert cm.ndim == 3

        cm.phases = phase_list

        cm[0, :2, :1].phase_id = 1
        cm[1, 2:4, :1].phase_id = 2
        assert np.allclose(np.unique(cm.phase_id), np.array([0, 1, 2]))

        # One phase plot per masked map
        for i, phase in phase_list:
            fig = plt.figure()
            ax = fig.add_subplot(projection=PLOT_MAP)
            _ = ax.plot_map(cm[cm.phase_id == i])

        plt.close("all")

    @pytest.mark.parametrize(
        "crystal_map_input",
        [((2, 9, 6), (1, 1.5, 1.5), 2, [0]), ((2, 10, 5), (1, 0.1, 0.1), 1, [0])],
        indirect=["crystal_map_input"],
    )
    @pytest.mark.filterwarnings(f"ignore: {SCALEBAR_WARNING}")
    def test_plot_masked_scalar(self, crystal_map_input):
        cm = CrystalMap(**crystal_map_input)

        cm.prop["test"] = np.zeros(cm.size)
        cm[1, 5, 4].test = 1

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection=PLOT_MAP)
        im1 = ax1.plot_map(cm, cm.test)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(projection=PLOT_MAP)
        im2 = ax2.plot_map(cm, cm.test, depth=1)

        expected_data_im1 = np.zeros(ax1._data_shape)
        expected_data_im2 = np.zeros(ax2._data_shape)
        expected_data_im2[5, 4] = 1

        im1_data = im1.get_array()
        im2_data = im2.get_array()
        assert np.allclose(im1_data, expected_data_im1)
        assert np.allclose(im2_data, expected_data_im2)

        plt.close("all")


class TestCrystalMapPlotUtilities:
    def test_init_projection(self):
        # Option 1
        fig = plt.figure()
        ax1 = fig.add_subplot(projection=PLOT_MAP)
        assert isinstance(ax1, CrystalMapPlot)

        # Option 2 (`label` to suppress warning of non-unique figure objects)
        ax2 = plt.subplot(projection=PLOT_MAP, label="unique")
        assert isinstance(ax2, CrystalMapPlot)

        plt.close("all")

    @pytest.mark.parametrize(
        "crystal_map_input, idx_to_change, axes, depth, expected_plot_shape",
        [
            (((2, 10, 20), (1, 1.5, 1.5), 1, [0]), (1, 5, 4), (1, 2), 1, (10, 20)),
            (((4, 4, 3), (0.1, 0.1, 0.1), 1, [0]), (0, 0, 2), (0, 1), 2, (4, 4)),
            (((10, 10, 10), (1, 1, 1), 2, [0]), (-1, 8, -1), (0, 2), 8, (10, 10)),
        ],
        indirect=["crystal_map_input"],
    )
    @pytest.mark.filterwarnings(f"ignore: {SCALEBAR_WARNING}")
    def test_get_plot_shape(
        self, crystal_map_input, idx_to_change, axes, depth, expected_plot_shape
    ):
        cm = CrystalMap(**crystal_map_input)

        cm.prop["test"] = np.zeros(cm.size)
        cm[idx_to_change].test = 1

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        im = ax.plot_map(cm, cm.test, axes=axes, depth=depth)

        assert ax._data_shape == expected_plot_shape
        assert np.allclose(np.unique(im.get_array()), np.array([0, 1]))

        plt.close("all")

    @pytest.mark.parametrize("to_plot", ["scalar", "rgb"])
    @pytest.mark.filterwarnings(f"ignore: {SCALEBAR_WARNING}")
    def test_add_overlay(self, crystal_map, to_plot):
        cm = crystal_map

        assert np.allclose(cm.phase_id, np.zeros(cm.size))  # Assumption

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        if to_plot == "scalar":
            im = ax.plot_map(cm, cm.y)
            im_data = im.get_array()
            assert im_data.ndim == 2
            im_data = im.to_rgba(im_data)[:, :, :3]
        else:  # rgb
            im = ax.plot_map(cm)
            im_data = copy.deepcopy(im.get_array())

        to_overlay = cm.id
        ax.add_overlay(cm, to_overlay)
        im_data2 = ax.images[0].get_array()

        assert np.allclose(im_data, im_data2) is False

        overlay = cm.get_map_data(to_overlay)
        overlay_min = np.nanmin(overlay)
        rescaled_overlay = (overlay - overlay_min) / (np.nanmax(overlay) - overlay_min)

        for i in range(3):
            im_data[:, :, i] *= rescaled_overlay

        assert np.allclose(im_data, im_data2)

        plt.close("all")

    @pytest.mark.parametrize(
        "phase_names, phase_colors, legend_properties",
        [
            (["a", "b"], ["r", "b"], {}),
            (
                ["austenite", "ferrite"],
                ["xkcd:violet", "lime"],
                {"borderpad": 1, "framealpha": 1},
            ),
            (
                ["al", "au"],
                ["tab:orange", "tab:green"],
                {"handlelength": 0.5, "handletextpad": 0.5},
            ),
        ],
    )
    @pytest.mark.filterwarnings(f"ignore: {SCALEBAR_WARNING}")
    def test_phase_legend(
        self, crystal_map, phase_names, phase_colors, legend_properties
    ):
        cm = crystal_map
        cm[0, 0].phase_id = 1
        cm.phases = PhaseList(
            names=phase_names, point_groups=[3, 3], colors=phase_colors
        )

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        _ = ax.plot_map(cm, legend_properties=legend_properties)

        legend = ax.legend_

        assert legend._fontsize == 11.0
        assert [i._text for i in legend.texts] == phase_names

        frame_alpha = legend_properties.pop("framealpha", 0.6)
        assert legend.get_frame().get_alpha() == frame_alpha

        for k, v in legend_properties.items():
            assert legend.__getattribute__(k) == v

        plt.close("all")

    @pytest.mark.parametrize("with_colorbar", [True, False])
    @pytest.mark.filterwarnings(f"ignore: {SCALEBAR_WARNING}")
    def test_remove_padding(self, crystal_map, with_colorbar):
        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        _ = ax.plot_map(crystal_map)

        if with_colorbar:
            _ = ax.add_colorbar(position="right")

        # Before
        margin_before = 0.05
        assert ax._xmargin == margin_before
        assert ax._ymargin == margin_before

        expected_subplot_params = (0.88, 0.11, 0.9, 0.125)  # top, bottom, right, left
        subplot_params = fig.subplotpars
        assert (
            subplot_params.top,
            subplot_params.bottom,
            subplot_params.right,
            subplot_params.left,
        ) == expected_subplot_params

        ax.remove_padding()

        # After
        margin_after = 0
        expected_subplot_params = (1, 0, 1, 0)
        assert ax._xmargin == margin_after
        assert ax._ymargin == margin_after

        if with_colorbar:
            expected_subplot_params = (1, 0, 0.9, 0)
        subplot_params = fig.subplotpars
        assert (
            subplot_params.top,
            subplot_params.bottom,
            subplot_params.right,
            subplot_params.left,
        ) == expected_subplot_params

        plt.close("all")

    @pytest.mark.parametrize("cmap", ["viridis", "cividis", "inferno"])
    @pytest.mark.filterwarnings(f"ignore: {SCALEBAR_WARNING}")
    def test_set_colormap(self, crystal_map, cmap):
        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        im = ax.plot_map(crystal_map, cmap=cmap)

        assert im.cmap.name == cmap

        plt.close("all")

    @pytest.mark.parametrize(
        "cmap, label, position",
        [("viridis", "a", "right"), ("cividis", "b", "left"), ("inferno", "c", "top"),],
    )
    @pytest.mark.filterwarnings(f"ignore: {SCALEBAR_WARNING}")
    def test_add_colorbar(self, crystal_map, cmap, label, position):
        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        im = ax.plot_map(crystal_map, cmap=cmap)

        assert im.cmap.name == cmap

        cbar = ax.add_colorbar(label=label, position=position)

        assert cbar.cmap.name == cmap
        assert cbar.ax.get_ylabel() == label

        new_label = label + "z"
        cbar.ax.set_ylabel(ylabel=new_label)
        assert cbar.ax.get_ylabel() == new_label

        assert cbar.ax.yaxis.labelpad == 15

        plt.close("all")

    @pytest.mark.parametrize(
        "value, unit, expected_value, expected_unit, expected_factor",
        [
            (3.141592 * 1e3, "nm", 3.141592, "um", 1e3),
            (10.55 * 1e-3, "mm", 10.55, "um", 1e-3),
            (23.7 * 1e3, "px", 23.7, "px", 1e3),
        ],
    )
    def test_convert_unit(
        self, value, unit, expected_value, expected_unit, expected_factor
    ):
        new_value, new_unit, factor = convert_unit(value, unit)
        assert np.allclose(new_value, expected_value, atol=1e-6)
        assert new_unit == expected_unit
        assert np.allclose(factor, expected_factor)


class TestStatusBar:
    @pytest.mark.filterwarnings(f"ignore: {SCALEBAR_WARNING}")
    def test_status_bar_call_directly(self, crystal_map):
        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        im = ax.plot_map(crystal_map)

        _ = ax._override_status_bar(im, crystal_map)

        plt.close("all")

    @pytest.mark.filterwarnings(f"ignore: {SCALEBAR_WARNING}")
    def test_status_bar_silence_default_format_coord(self, crystal_map):
        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        _ = ax.plot_map(crystal_map)

        assert ax.format_coord(0, 0) == ""

        plt.close("all")

    @pytest.mark.parametrize("to_plot", ["rgb", "scalar"])
    @pytest.mark.filterwarnings(f"ignore: {SCALEBAR_WARNING}")
    def test_status_bar(self, crystal_map, to_plot):
        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)

        f = plt.gcf()
        f.canvas.draw()
        f.canvas.flush_events()

        if to_plot == "rgb":
            im = ax.plot_map(crystal_map)
        else:  # scalar
            im = ax.plot_map(crystal_map, crystal_map.id)

        # Get figure canvas (x, y) from transformation from data (x, y)
        data_idx = (2, 2)
        x, y = ax.transData.transform(data_idx)

        # Mock a mouse event
        plt.matplotlib.backends.backend_agg.FigureCanvasBase.motion_notify_event(
            f.canvas, x, y
        )
        cursor_event = plt.matplotlib.backend_bases.MouseEvent(
            "motion_notify_event", f.canvas, x, y
        )

        # Call our custom cursor data function
        cursor_data = im.get_cursor_data(cursor_event)

        # Check status bar data formatting
        data_format = im.format_cursor_data(cursor_data)
        expected_format = f"(y,x):({data_idx[0]},{data_idx[1]})"

        # Get crystal map data
        point = crystal_map[data_idx]
        r = point.rotations.to_euler()

        # Expected indices, rotation and value
        assert cursor_data[:2] == data_idx
        assert np.allclose(cursor_data[2], r, atol=1e-3)
        if to_plot == "rgb":
            assert np.allclose(cursor_data[3], point.phases_in_data.colors_rgb)
        else:  # scalar
            assert np.allclose(cursor_data[3], point.id)
            expected_format += " val: {:.1f}".format(point.id[0])

        expected_format += " rot:({:.3f},{:.3f},{:.3f})".format(
            r[0, 0], r[0, 1], r[0, 2]
        )
        assert data_format == expected_format

        plt.close("all")


class TestScalebar:
    @pytest.mark.parametrize(
        "crystal_map_input, scalebar_properties, artist_attribute",
        [
            (((1, 10, 30), (0, 1, 1), 1, [0]), {}, {}),  # Default
            (
                ((1, 10, 30), (0, 1, 1), 1, [0]),
                {"loc": 4, "sep": 6, "size_vertical": 0.2, "alpha": 0.8,},
                {
                    "loc": "loc",
                    "sep": ["_box", "sep"],
                    "size_vertical": ["size_bar", "_children", 0, "_height"],
                    "alpha": ["patch", "_alpha"],
                },
            ),
        ],
        indirect=["crystal_map_input"],
    )
    def test_scalebar_properties(
        self, crystal_map_input, scalebar_properties, artist_attribute
    ):
        cm = CrystalMap(**crystal_map_input)
        cm.scan_unit = "um"

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        _ = ax.plot_map(cm, scalebar=False)
        sbar = ax.add_scalebar(cm, **scalebar_properties)

        if "alpha" not in scalebar_properties.keys():  # Check default
            assert sbar.patch._alpha == 0.6

        for (k, v), attr_loc in zip(
            scalebar_properties.items(), artist_attribute.values()
        ):
            if isinstance(attr_loc, list):
                sbar_attr = sbar.__getattribute__(attr_loc[0])
                for i in attr_loc[1:]:
                    if isinstance(i, int):
                        sbar_attr = sbar_attr[i]
                    else:
                        sbar_attr = sbar_attr.__getattribute__(i)
            else:
                sbar_attr = sbar.__getattribute__(attr_loc)
            assert sbar_attr == v

        plt.close("all")

    @pytest.mark.parametrize(
        (
            "crystal_map_input, unit, expected_coordinate_axes, expected_width_px, "
            "expected_width_unit, expected_unit"
        ),
        [
            (((10, 20, 1), (1, 1, 0), 1, [0]), "px", {0: "z", 1: "y"}, 2, 2, "px"),
            (((1, 10, 30), (0, 1, 1), 1, [0]), "px", {0: "y", 1: "x"}, 2, 2, "px"),
            (((1, 10, 40), (0, 1, 1), 1, [0]), "px", {0: "y", 1: "x"}, 5, 5, "px"),
            (((20, 1, 10), (1, 1, 1), 1, [0]), "um", {0: "z", 1: "x"}, 1, 1, "um"),
            (
                ((1, 100, 117), (0, 1.5, 1.5), 1, [0]),
                "nm",
                {0: "y", 1: "x"},
                13.33,
                20,
                "nm",
            ),
        ],
        indirect=["crystal_map_input"],
    )
    def test_scalebar_axis(
        self,
        crystal_map_input,
        unit,
        expected_coordinate_axes,
        expected_width_px,
        expected_width_unit,
        expected_unit,
    ):
        cm = CrystalMap(**crystal_map_input)
        cm.scan_unit = unit

        assert cm._coordinate_axes == expected_coordinate_axes

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        _ = ax.plot_map(cm, scalebar=False)
        sbar = ax.add_scalebar(cm)

        # Scalebar width (_width attribute of Rectangle artist)
        assert np.allclose(
            sbar.size_bar._children[0]._width, expected_width_px, atol=1e-2
        )

        # Scalebar text
        if expected_unit == "um":
            expected_unit = "\u03BC" + "m"
        assert (
            sbar.txt_label._children[0]._text
            == f"{expected_width_unit} {expected_unit}"
        )

        plt.close("all")

    @pytest.mark.parametrize(
        "crystal_map_input, warns",
        [
            (((1, 10, 20), (0, 1.5, 1.5), 1, [0]), False),
            (((1, 4, 3), (0, 0.1, 0.1), 1, [0]), True),
        ],
        indirect=["crystal_map_input"],
    )
    def test_scalebar_warns(self, crystal_map_input, warns):
        cm = CrystalMap(**crystal_map_input)

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)

        if warns:
            with pytest.warns(UserWarning, match=SCALEBAR_WARNING):
                _ = ax.plot_map(cm)
        else:
            _ = ax.plot_map(cm)

        plt.close("all")
