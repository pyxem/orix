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

from typing import Optional, Tuple, Union

from matplotlib import projections
from matplotlib.gridspec import SubplotSpec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from orix.vector import AxAngle, Rodrigues


class RotationPlot(Axes3D):
    """Plot of a (mis)orientation region."""

    name = None
    transformation_class = None

    def transform(
        self,
        xs: Union["Misorientation", "OrientationRegion", "Rotation"],
        fundamental_zone: Optional["OrientationRegion"] = None,
    ):
        """Prepare (mis)orientations or rotations for plotting.

        Parameters
        ----------
        xs
            Object to transform.
        fundamental_zone
            Orientation region to add to the plot as a wireframe.
        """
        from orix.quaternion import Misorientation, OrientationRegion, Rotation

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
            if isinstance(xs, OrientationRegion):
                xs = xs.get_plot_data()
            transformed = self.transformation_class.from_rotation(xs)
        else:
            transformed = self.transformation_class(xs)
        x, y, z = transformed.xyz
        return x, y, z

    def scatter(
        self, xs: Union["Misorientation", "Rotation"], fundamental_zone=None, **kwargs
    ):
        """Create a scatter plot.

        Parameters
        ----------
        xs
            Rotations.
        fundamental_zone
            Orientation region to add to the plot as a wireframe.
        **kwargs
            Keyword arguments passed to
            :meth:`mpl_toolkits.mplot3d.Axes3D.scatter`.
        """
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

    def _get_region_extent(
        self, fundamental_region: "OrientationRegion"
    ) -> Tuple[float, float, float]:
        """Return the maximum angles in x, y, z of the fundamental
        region.

        Parameters
        ----------
        fundamental_region

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
    """Plot rotations in Rodrigues-Frank space."""

    name = "rodrigues"
    transformation_class = Rodrigues


class AxAnglePlot(RotationPlot):
    """Plot rotations in a axis-angle space."""

    name = "axangle"
    transformation_class = AxAngle


projections.register_projection(RodriguesPlot)
projections.register_projection(AxAnglePlot)


def _setup_rotation_plot(
    figure: Optional[plt.Figure] = None,
    projection: str = "axangle",
    position: Union[int, tuple, SubplotSpec, None] = (1, 1, 1),
    figure_kwargs: Optional[dict] = None,
) -> Tuple[plt.Figure, Union[AxAnglePlot, RodriguesPlot]]:
    """Return a figure and rotation plot axis of the correct type.

    This is a convenience method used in e.g.
    :meth:`orix.quaternion.Orientation.scatter`.

    Parameters
    ----------
    figure
        If given, a new plot axis :class:`orix.plot.AxAnglePlot` or
        :class:`orix.plot.RodriguesPlot` is added to the figure in
        the position specified by `position`. If not given, a new
        figure is created.
    projection
        Which orientation space to plot orientations in, either
        "axangle" (default) or "rodrigues".
    position
        Where to add the new plot axis. 121 or (1, 2, 1) places it
        in the first of two positions in a grid of 1 row and 2
        columns. See :meth:`matplotlib.figure.Figure.add_subplot`
        for further details. Default is (1, 1, 1).
    figure_kwargs
        Dictionary of keyword arguments passed to
        :func:`matplotlib.pyplot.figure` if `figure` is not given.

    Returns
    -------
    figure
        Figure with the added plot axis.
    ax
        The axis-angle or Rodrigues plot axis.
    """
    if figure is None:
        if figure_kwargs is None:
            figure_kwargs = {"layout": "tight"}
        figure = plt.figure(**figure_kwargs)

    subplot_kwds = {
        "projection": projection,
        "proj_type": "ortho",
        "auto_add_to_figure": False,
    }

    if hasattr(position, "__iter__"):
        # (nrows, ncols, index)
        ax = figure.add_subplot(*position, **subplot_kwds)
    else:
        # Position, e.g. 121, or `SubplotSpec`
        ax = figure.add_subplot(position, **subplot_kwds)

    return figure, ax
