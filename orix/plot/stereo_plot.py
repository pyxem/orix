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

from matplotlib.axes import Axes
from matplotlib.patches import Circle
from orix.vector.vector3d import Vector3d
from orix.projections.stereographic_projection import StereographicProjection
from matplotlib.projections import register_projection


class StereoPlot(Axes):
    """Plot Vector3D in a stereographic plot"""

    name = "stereographic"

    def scatter(self, vectors3d, **kwargs):
        xy = StereographicProjection.project(vectors3d)
        self.add_patch(Circle((0, 0), 1, facecolor="none", edgecolor="black"))
        self.set_aspect("equal")
        self.set_xlim(-1.1, 1.1)
        self.set_ylim(-1.1, 1.1)
        return super().scatter(xy[:, 0], xy[:, 1], **kwargs)

    def plot(self, vectors3d, **kwargs):
        return self.scatter(vectors3d, **kwargs)


register_projection(StereoPlot)
