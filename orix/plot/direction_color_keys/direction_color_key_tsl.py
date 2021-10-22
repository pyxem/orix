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

import numpy as np
from scipy.interpolate import griddata

from orix.plot.direction_color_keys import DirectionColorKey
from orix.plot.direction_color_keys._util import (
    polar_coordinates_in_sector,
    rgb_from_polar_coordinates,
)
from orix.projections import StereographicProjection
from orix.sampling import sample_S2_cube_mesh
from orix.vector import Vector3d


class DirectionColorKeyTSL(DirectionColorKey):
    def __init__(self, symmetry):
        laue_symmetry = symmetry.laue
        super().__init__(laue_symmetry)

    def direction2color(self, direction):
        laue_group = self.symmetry
        h = direction.in_fundamental_sector(laue_group)
        azimuth, polar = polar_coordinates_in_sector(laue_group.fundamental_sector, h)
        polar = 0.5 + polar / 2
        v = Vector3d.from_polar(azimuth=azimuth, polar=polar * np.pi)
        return rgb_from_polar_coordinates(v.azimuth.data, v.polar.data)

    def plot(self, return_figure=False):
        from orix.plot.inverse_pole_figure_plot import _setup_inverse_pole_figure_plot

        laue_group = self.symmetry
        sector = laue_group.fundamental_sector

        # Set S2 sampling and color grid interpolation resolutions
        resolution_s2 = 2
        resolution_grid = 0.001

        v = sample_S2_cube_mesh(resolution_s2)
        v2 = Vector3d(np.row_stack((v[v <= sector].data, sector.edges.data)))

        rgb = self.direction2color(v2)
        r, g, b = rgb.T

        x, y = StereographicProjection().vector2xy(v2)
        yx = np.column_stack((y, x))

        y_min = np.min(y)
        y_max = np.max(y)
        x_min = np.min(x)
        x_max = np.max(x)
        grid_y, grid_x = np.mgrid[
            y_min:y_max:resolution_grid, x_min:x_max:resolution_grid
        ]

        method = "cubic"
        r2 = griddata(yx, r, (grid_y, grid_x), method=method)
        g2 = griddata(yx, g, (grid_y, grid_x), method=method)
        b2 = griddata(yx, b, (grid_y, grid_x), method=method)
        rgb_grid = np.clip(np.dstack((r2, g2, b2)), 0, 1)
        rgb_grid[np.isnan(r2)] = 1
        rgb_grid = np.flipud(rgb_grid)

        fig, axes = _setup_inverse_pole_figure_plot(laue_group)
        ax = axes[0]
        for loc in ["left", "center", "right"]:
            title = ax.get_title(loc)
            if title != "":
                ax.set_title(laue_group.name, loc=loc, fontweight="bold")
        ax.stereographic_grid(False)
        ax._edge_patch.set_linewidth(1.5)
        ax.imshow(rgb_grid, extent=(x_min, x_max, y_min, y_max), zorder=0)

        if return_figure:
            return fig
