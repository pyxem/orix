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

from matplotlib import projections
from mpl_toolkits.mplot3d import Axes3D
from orix.vector.neo_euler import Rodrigues, AxAngle


class RotationPlot(Axes3D):

    name = None
    transformation_class = None

    def transform(self, xs):
        from orix.quaternion.rotation import Rotation

        if isinstance(xs, Rotation):
            transformed = self.transformation_class.from_rotation(xs.get_plot_data())
        else:
            transformed = self.transformation_class(xs)
        x, y, z = transformed.xyz
        return x, y, z

    def scatter(self, xs, **kwargs):
        x, y, z = self.transform(xs)
        return super().scatter(x, y, z, **kwargs)

    def plot(self, xs, **kwargs):
        x, y, z = self.transform(xs)
        return super().plot(x, y, z, **kwargs)

    def plot_wireframe(self, xs, **kwargs):
        x, y, z = self.transform(xs)
        return super().plot_wireframe(x, y, z, **kwargs)


class RodriguesPlot(RotationPlot):
    """Plot rotations in a Rodrigues-Frank projection."""

    name = "rodrigues"
    transformation_class = Rodrigues


class AxAnglePlot(RotationPlot):
    """Plot rotations in an Axes-Angle projection."""

    name = "axangle"
    transformation_class = AxAngle


projections.register_projection(RodriguesPlot)
projections.register_projection(AxAnglePlot)
