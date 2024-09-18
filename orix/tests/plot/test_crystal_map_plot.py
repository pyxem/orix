# Copyright 2018-2024 the orix developers
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

import matplotlib.colorbar as mbar
import matplotlib.pyplot as plt
from matplotlib_scalebar import scalebar
import numpy as np
import pytest

from orix.crystal_map import CrystalMap, PhaseList
from orix.plot import CrystalMapPlot

# Can be easily changed in the future
PLOT_MAP = "plot_map"


class TestCrystalMapPlot:
    @pytest.mark.parametrize(
        "crystal_map_input, expected_data_shape",
        [
            (((10, 20), (1.5, 1.5), 1, [0]), (10, 20, 3)),
            (((4, 3), (0.1, 0.1), 1, [0]), (4, 3, 3)),
        ],
        indirect=["crystal_map_input"],
    )
    def test_plot_phase(self, crystal_map_input, phase_list, expected_data_shape):
        xmap = CrystalMap(**crystal_map_input)

        assert np.unique(xmap.phase_id) == np.array([0])  # Test code assumption
        assert phase_list.ids == [0, 1, 2]
        xmap.phases = phase_list
        xmap[0, 0].phase_id = 0
        xmap[1, 1].phase_id = 2

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        im = ax.plot_map(xmap)

        # Expected image data
        phase_id = xmap.get_map_data("phase_id")
        unique_phase_ids = np.unique(phase_id[~np.isnan(phase_id)])
        expected_data = np.ones(phase_id.shape + (3,))
        for i, color in zip(unique_phase_ids, xmap.phases_in_data.colors_rgb):
            mask = phase_id == int(i)
            expected_data[mask] *= color

        image_data = im.get_array()
        assert np.allclose(image_data.shape, expected_data_shape)
        assert np.allclose(image_data, expected_data)

        plt.close("all")

    def test_plot_property(self, crystal_map):
        xmap = crystal_map

        prop_name = "iq"
        prop_data = np.arange(xmap.size)
        xmap.prop[prop_name] = prop_data

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        im = ax.plot_map(xmap, xmap.iq)

        assert np.allclose(im.get_array(), prop_data.reshape(xmap.shape))

        plt.close("all")

    def test_plot_scalar(self, crystal_map):
        xmap = crystal_map

        angles = xmap.rotations.angle

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection=PLOT_MAP)
        # Test use of `vmax` as well
        im1 = ax1.plot_map(xmap, angles, vmax=angles.max() - 10)

        assert np.allclose(im1.get_array(), angles.reshape(xmap.shape), atol=1e-3)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(projection=PLOT_MAP)
        im2 = ax2.plot_map(xmap, xmap.rotations.angle)

        assert np.allclose(im2.get_array(), angles.reshape(xmap.shape), atol=1e-3)

        plt.close("all")

    @pytest.mark.parametrize(
        "crystal_map_input",
        [((9, 3), (1.5, 1.5), 1, [0]), ((10, 5), (0.1, 0.1), 1, [0])],
        indirect=["crystal_map_input"],
    )
    def test_plot_masked_phase(self, crystal_map_input, phase_list):
        xmap = CrystalMap(**crystal_map_input)

        # Test code assumptions
        assert np.unique(xmap.phase_id) == np.array([0])
        assert phase_list.ids == [0, 1, 2]
        assert xmap.ndim == 2

        xmap.phases = phase_list

        xmap[:2, :1].phase_id = 1
        xmap[2:4, :1].phase_id = 2
        assert np.allclose(np.unique(xmap.phase_id), np.array([0, 1, 2]))

        # One phase plot per masked map
        for i, phase in phase_list:
            fig = plt.figure()
            ax = fig.add_subplot(projection=PLOT_MAP)
            _ = ax.plot_map(xmap[xmap.phase_id == i])

        plt.close("all")

    @pytest.mark.parametrize(
        "crystal_map_input",
        [((9, 6), (1.5, 1.5), 2, [0]), ((10, 5), (0.1, 0.1), 1, [0])],
        indirect=["crystal_map_input"],
    )
    def test_plot_masked_scalar(self, crystal_map_input):
        xmap = CrystalMap(**crystal_map_input)

        xmap.prop["test"] = np.zeros(xmap.size)
        xmap[5, 4].test = 1

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        im = ax.plot_map(xmap, xmap.test)

        expected_data_im = np.zeros(ax._data_shape)
        expected_data_im[5, 4] = 1

        im_data = im.get_array()
        assert np.allclose(im_data, expected_data_im)

        plt.close("all")

    def test_properties(self, crystal_map):
        xmap = crystal_map
        fig = xmap.plot(return_figure=True, colorbar=True, colorbar_label="score")
        ax = fig.axes[0]

        assert isinstance(ax.colorbar, mbar.Colorbar)
        assert ax.colorbar.ax.get_ylabel() == "score"
        assert isinstance(ax.scalebar, scalebar.ScaleBar)

        plt.close("all")


class TestCrystalMapPlotUtilities:
    def test_init_projection(self):
        # Option 1
        _ = plt.figure()
        ax1 = plt.subplot(projection=PLOT_MAP)
        assert isinstance(ax1, CrystalMapPlot)

        # Option 2 (recommended)
        fig = plt.figure()
        ax2 = fig.add_subplot(projection=PLOT_MAP)
        assert isinstance(ax2, CrystalMapPlot)

        plt.close("all")

    @pytest.mark.parametrize(
        "crystal_map_input, idx_to_change, expected_plot_shape",
        [
            (((10, 20), (1.5, 1.5), 1, [0]), (5, 4), (10, 20)),
            (((4, 3), (0.1, 0.1), 1, [0]), (0, 2), (4, 3)),
            (((10, 10), (1, 1), 2, [0]), (8, -1), (10, 10)),
        ],
        indirect=["crystal_map_input"],
    )
    def test_get_plot_shape(
        self, crystal_map_input, idx_to_change, expected_plot_shape
    ):
        xmap = CrystalMap(**crystal_map_input)

        xmap.prop["test"] = np.zeros(xmap.size)
        xmap[idx_to_change].test = 1

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        im = ax.plot_map(xmap, xmap.test)

        assert ax._data_shape == expected_plot_shape
        assert np.allclose(np.unique(im.get_array()), np.array([0, 1]))

        plt.close("all")

    @pytest.mark.parametrize("to_plot", ["scalar", "rgb"])
    def test_add_overlay(self, crystal_map, to_plot):
        xmap = crystal_map

        assert np.allclose(xmap.phase_id, np.zeros(xmap.size))  # Assumption

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        if to_plot == "scalar":
            im = ax.plot_map(xmap, xmap.y)
            im_data = im.get_array()
            assert im_data.ndim == 2
            im_data = im.to_rgba(im_data)[:, :, :3]
        else:  # rgb
            im = ax.plot_map(xmap)
            im_data = copy.deepcopy(im.get_array())

        to_overlay = xmap.id
        ax.add_overlay(xmap, to_overlay)
        im_data2 = ax.images[0].get_array()

        assert np.allclose(im_data, im_data2) is False

        overlay = xmap.get_map_data(to_overlay)
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
    def test_phase_legend(
        self, crystal_map, phase_names, phase_colors, legend_properties
    ):
        xmap = crystal_map
        xmap[0, 0].phase_id = 1
        xmap.phases = PhaseList(
            names=phase_names, point_groups=[3, 3], colors=phase_colors
        )

        fontsize = 11
        plt.rcParams["font.size"] = fontsize

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        _ = ax.plot_map(xmap, legend_properties=legend_properties)

        legend = ax.legend_

        assert legend._fontsize == fontsize
        assert [i._text for i in legend.texts] == phase_names

        frame_alpha = legend_properties.pop("framealpha", 0.6)
        assert legend.get_frame().get_alpha() == frame_alpha

        for k, v in legend_properties.items():
            assert legend.__getattribute__(k) == v

        plt.close("all")

    @pytest.mark.parametrize("with_colorbar", [True, False])
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
    def test_set_colormap(self, crystal_map, cmap):
        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        im = ax.plot_map(crystal_map, cmap=cmap)

        assert im.cmap.name == cmap

        plt.close("all")

    @pytest.mark.parametrize(
        "cmap, label, position",
        [("viridis", "a", "right"), ("cividis", "b", "left"), ("inferno", "c", "top")],
    )
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


class TestStatusBar:
    def test_status_bar_call_directly(self, crystal_map):
        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        im = ax.plot_map(crystal_map)

        _ = ax._override_status_bar(im, crystal_map)

        plt.close("all")

    def test_status_bar_silence_default_format_coord(self, crystal_map):
        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        _ = ax.plot_map(crystal_map)
        # TODO: Remove (x, y) = (0, 0) when requuring matplotlib > 3.8
        assert ax.format_coord(0, 0) in ["(x, y) = (0, 0)", "x=0 y=0"]

        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)
        _ = ax.plot_map(crystal_map, override_status_bar=True)
        assert ax.format_coord(0, 0) == ""

        plt.close("all")

    @pytest.mark.parametrize("to_plot", ["rgb", "scalar"])
    def test_status_bar(self, crystal_map, to_plot):
        fig = plt.figure()
        ax = fig.add_subplot(projection=PLOT_MAP)

        f = plt.gcf()
        f.canvas.draw()
        f.canvas.flush_events()

        if to_plot == "rgb":
            im = ax.plot_map(crystal_map, override_status_bar=True)
        else:  # scalar
            im = ax.plot_map(crystal_map, crystal_map.id, override_status_bar=True)

        # Get figure canvas (x, y) from transformation from data (x, y)
        data_idx = (2, 2)
        x, y = ax.transData.transform(data_idx)

        # Mock a mouse event
        cursor_event = plt.matplotlib.backend_bases.MouseEvent(
            "motion_notify_event", f.canvas, x, y
        )
        f.canvas.callbacks.process(cursor_event.name, cursor_event)

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
        "crystal_map_input, scalebar_properties",
        [
            (((10, 30), (1, 1), 1, [0]), {}),  # Default
            (
                ((10, 30), (1, 1), 1, [0]),
                {"location": 4, "sep": 6, "box_alpha": 0.8},
            ),
        ],
        indirect=["crystal_map_input"],
    )
    def test_scalebar_properties(self, crystal_map_input, scalebar_properties):
        xmap = CrystalMap(**crystal_map_input)
        units = "um"
        xmap.scan_unit = units
        fig = xmap.plot(return_figure=True, scalebar=False)
        sbar = fig.axes[0].add_scalebar(xmap, **scalebar_properties)
        assert sbar.units == units
        assert sbar.dx == xmap.dx
        for k, v in scalebar_properties.items():
            assert sbar.__getattribute__(k) == v

        # Custom scan unit
        xmap.scan_unit = "parsec"
        fig2 = xmap.plot(return_figure=True)
        assert fig2.axes[0].artists[0].units == xmap.scan_unit

        plt.close("all")
