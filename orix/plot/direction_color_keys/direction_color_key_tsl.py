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

from orix.plot.direction_color_keys import DirectionColorKey
from orix.plot.direction_color_keys._util import polar_coordinates, polar2rgb
from orix.sampling import uniform_S2_sample
from orix.vector import Vector3d


class DirectionColorKeyTSL(DirectionColorKey):
    def __init__(self, symmetry):
        laue_symmetry = symmetry.laue
        super().__init__(laue_symmetry)

    def direction2color(self, direction):
        laue_group = self.symmetry

        m = direction.in_fundamental_sector(laue_group)

        sector = laue_group.fundamental_sector
        center = sector.center

        vertex = sector.vertices[0]

        azimuth, radius = polar_coordinates(sector, m, center, vertex)
        radius = 0.5 + radius / 2
        v = Vector3d.from_polar(azimuth=azimuth, polar=radius * np.pi)
        rgb = polar2rgb(v.azimuth.data, v.polar.data)

        return rgb

    def plot(self, return_figure=False):
        from orix.plot.inverse_pole_figure_plot import _setup_inverse_pole_figure_plot

        laue_group = self.symmetry
        v = uniform_S2_sample(0.75)

        sector = laue_group.fundamental_sector
        v2 = v[v < sector]
        rgb = self.direction2color(v2)

        fig, axes = _setup_inverse_pole_figure_plot(laue_group)
        ax = axes[0]
        for loc in ["left", "center", "right"]:
            title = ax.get_title(loc)
            if title != "":
                ax.set_title(laue_group.name, loc=loc, fontweight="bold")
        ax.stereographic_grid(False)
        ax.scatter(v2, c=rgb, zorder=1, clip_on=True)

        if return_figure:
            return fig
