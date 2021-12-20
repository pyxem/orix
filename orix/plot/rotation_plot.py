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

from matplotlib import projections, __version__
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from orix.vector import AxAngle, Rodrigues


class RotationPlot(Axes3D):

    name = None
    transformation_class = None

    def transform(self, xs, fundamental_zone=None):
        from orix.quaternion import Rotation, Misorientation, OrientationRegion

        # Project rotations into fundamental zone if necessary
        if isinstance(xs, Misorientation):
            if fundamental_zone is None:
                symmetry = xs.symmetry
                # Orientation.symmetry returns a Symmetry object not a tuple, so pack
                if not isinstance(symmetry, tuple):
                    symmetry = (symmetry,)
                fundamental_zone = OrientationRegion.from_symmetry(*symmetry)
            # check fundamental_zone is properly defined
            if not isinstance(fundamental_zone, OrientationRegion):
                raise TypeError("fundamental_zone is not an OrientationRegion object.")
            # if any in xs are out of fundamental_zone, calculate symmetry reduction
            if not (xs < fundamental_zone).all():
                xs = xs.map_into_symmetry_reduced_zone()

        if isinstance(xs, Rotation):
            transformed = self.transformation_class.from_rotation(xs.get_plot_data())
        else:
            transformed = self.transformation_class(xs)
        x, y, z = transformed.xyz
        return x, y, z

    def scatter(self, xs, fundamental_zone=None, **kwargs):
        x, y, z = self.transform(xs, fundamental_zone=fundamental_zone)
        return super().scatter(x, y, z, **kwargs)

    def plot(self, xs, **kwargs):
        x, y, z = self.transform(xs)
        return super().plot(x, y, z, **kwargs)

    def plot_wireframe(self, xs, **kwargs):
        d = dict(color="gray", alpha=0.5, linewidth=0.5, rcount=30, ccount=30)
        for k, v in d.items():
            kwargs.setdefault(k, v)
        x, y, z = self.transform(xs)
        return super().plot_wireframe(x, y, z, **kwargs)

    def _get_region_extent(self, fundamental_region):
        """Return the maximum angles in x, y, z of the fundamental
        region.

        Parameters
        ----------
        fundamental_region : OrientationRegion

        Returns
        -------
        tuple of float
        """
        x, y, z = self.transform(fundamental_region)
        return x.max(), y.max(), z.max()

    def _correct_aspect_ratio(self, fundamental_region, set_limits=True):
        """Correct the aspect ratio of the axis according to the
        extent of the boundaries of the fundamental region.

        Parameters
        ----------
        fundamental_region : OrientationRegion
        set_limits : bool, optional
            Whether to also restrict the data limits to the boundary
            extents. Default is True.
        """
        xlim, ylim, zlim = self._get_region_extent(fundamental_region)
        self.set_box_aspect((xlim, ylim, zlim))
        if set_limits:
            self.set_xlim(-xlim, xlim)
            self.set_ylim(-ylim, ylim)
            self.set_zlim(-zlim, zlim)


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


def _setup_rotation_plot(figure=None, projection="axangle", position=None):
    """Return a figure and rotation plot axis of the correct type.

    This is a convenience method used in e.g.
    :meth:`orix.quaternion.Orientation.scatter`.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        If given, a new plot axis :class:`orix.plot.AxAnglePlot` or
        :class:`orix.plot.RodriguesPlot` is added to the figure in
        the position specified by `position`. If not given, a new
        figure is created.
    projection : str, optional
        Which orientation space to plot orientations in, either
        "axangle" (default) or "rodrigues".
    position : int, tuple of int, matplotlib.gridspec.SubplotSpec,
            optional
        Where to add the new plot axis. 121 or (1, 2, 1) places it
        in the first of two positions in a grid of 1 row and 2
        columns. See :meth:`matplotlib.figure.Figure.add_subplot`
        for further details. Default is (1, 1, 1).

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure with the added plot axis.
    ax : AxAnglePlot or RodriguesPlot
    """
    # Create figure if not already created, then add axis to figure
    if figure is None:
        figure = plt.figure()

    subplot_kwds = dict(projection=projection, proj_type="ortho")

    # TODO: Remove when the oldest supported version of Matplotlib increases
    # from 3.3 to 3.4.
    # See: https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html#mpl_toolkits.mplot3d.axes3d.Axes3D
    if version.parse(__version__) >= version.parse("3.4"):  # pragma: no cover
        subplot_kwds["auto_add_to_figure"] = False

    if position is None:
        position = (1, 1, 1)
    if hasattr(position, "__iter__"):
        # (nrows, ncols, index)
        ax = figure.add_subplot(*position, **subplot_kwds)
    else:
        # Position, e.g. 121, or `SubplotSpec`
        ax = figure.add_subplot(position, **subplot_kwds)

    return figure, ax
