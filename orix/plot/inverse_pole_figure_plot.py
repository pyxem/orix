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

"""Inverse pole figure plot inheriting from
:class:`~orix.plot.StereographicPlot` for plotting of
:class:`~orix.vector.Vector3d`, typically parallel to sample directions,
rotated by orientations.
"""

import matplotlib.axes as maxes
import matplotlib.pyplot as plt
import matplotlib.projections as mprojections
import numpy as np

from orix.crystal_map import Phase
from orix.plot import StereographicPlot
from orix.quaternion.symmetry import C1
from orix.vector import Miller, Vector3d


class InversePoleFigurePlot(StereographicPlot):
    """Inverse pole figure plot of :class:`~orix.vector.Vector3d`, which
    is a stereographic plot for showing sample directions with respect
    to a crystal reference frame.

    Inherits from :class:`~orix.plot.StereographicPlot`.
    """

    name = "ipf"

    def __init__(
        self,
        *args,
        symmetry=None,
        direction=None,
        hemisphere=None,
        **kwargs,
    ):
        """Create an inverse pole figure axis for plotting
        :class:`~orix.vector.Vector3d`.

        Parameters
        ----------
        args
            Arguments passed to
            :meth:`orix.plot.StereographicPlot.__init__`.
        symmetry : ~orix.quaternion.Symmetry, optional
            Laue group symmetry of crystal to plot directions with. If
            not given, point group C1 (only identity rotation) is used.
        direction : ~orix.vector.Vector3d, optional
            Sample direction to plot with respect to crystal directions.
            If not given, the out of plane direction, sample Z, is used.
        hemisphere : str, optional
            Which hemisphere(s) to plot the vectors in. If not given,
            "upper" is used. Options are "upper", "lower", and "both",
            which plots two projections side by side.
        hemisphere : str, optional
            Which hemisphere to plot vectors in, either "upper"
            (default) or "lower".
        kwargs
            Keyword arguments passed to
            :meth:`orix.plot.StereographicPlot.__init__`.
        """
        super().__init__(*args, **kwargs)

        if hemisphere is not None:
            self.hemisphere = hemisphere

        if direction is None:
            direction = Vector3d.zvector()
        self._direction = direction

        if symmetry is None:
            symmetry = C1
        self._symmetry = symmetry

        self.restrict_to_sector(self._symmetry.fundamental_sector)

        self._add_crystal_direction_labels()

    @property
    def _edge_patch(self):
        """Easy access to the fundamental sector border patch."""
        patches = self.patches
        return patches[self._has_collection(label="sa_sector", collections=patches)[1]]

    def scatter(self, *args, **kwargs):
        """A scatter plot of sample directions rotated by orientations,
        or orientations to rotate sample directions with.

        Parameters
        ----------
        args : tuple of numpy.ndarray, Orientation, or Vector3d
            Spherical coordinates (azimuth, polar), orientations, or
            vectors. If spherical coordinates are given, they are
            assumed to describe unit vectors. Vectors will be made into
            unit vectors if they aren't already. If orientations are
            passed, the crystal directions returned are the sample
            :attr:`direction` rotated by the orientations.
        kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.scatter`.

        See Also
        --------
        matplotlib.axes.Axes.scatter
        """
        crystal_directions = self._pretransform_input_ipf(args)
        super().scatter(crystal_directions, **kwargs)

    def show_hemisphere_label(self, **kwargs):
        """Add a hemisphere label ("upper"/"lower") to the upper left
        outside the inverse pole figure.

        Parameters
        ----------
        kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.text`.

        See Also
        --------
        hemisphere
        """
        new_kwargs = dict(ha="right", va="bottom")
        new_kwargs.update(kwargs)
        # TODO: Fix plotting of hemisphere labels for fundamental
        #  sectors with only two vertices on either side of equator (C3)
        x, y = self._edge_patch.get_path().vertices.T
        v = self._inverse_projection.xy2vector(np.min(x), np.max(y))
        self.text(v, s=self.hemisphere, **new_kwargs)

    def _add_crystal_direction_labels(self):
        """Add appropriately placed and nicely formatted crystal
        direction labels [uvw] or [UVTW] to the sector corners.
        """
        fs = self._symmetry.fundamental_sector
        vertices = fs.vertices
        if vertices.size > 0:
            center = fs.center.y.data[0]

            # Nicely formatted labels for crystal directions
            labels = _get_ipf_axes_labels(vertices, self._symmetry)
            x, y = self._projection.vector2xy(vertices)

            x_edge, y_edge = self._edge_patch.get_path().vertices.T
            x_min_edge, x_max_edge = np.min(x_edge), np.max(x_edge)
            y_min_edge, y_max_edge = np.min(y_edge), np.max(y_edge)
            pad = 0.01
            x_pad = pad * (x_max_edge - x_min_edge)
            y_pad = pad * (y_max_edge - y_min_edge)

            font_size = plt.rcParams["font.size"] + 4
            text_kw = dict(fontsize=font_size, zorder=10)
            for label, xi, yi in zip(labels, x, y):
                # Determine x and y coordinates of label relative to the
                # sector center, and adjust the alignment (starting
                # point) and shift accordingly, so that it doesn't cross
                # over the sector edge
                ha = "center"
                if np.isclose(yi, center, atol=1e-2):
                    va = "center"
                    if np.isclose(xi, x_min_edge, atol=1e-2):
                        ha = "right"
                        xi -= x_pad
                    else:
                        ha = "left"
                        xi += x_pad
                elif yi > center:
                    va = "bottom"
                    # Extra padding ensures [111] in symmetry Th is
                    # placed outside sector edge
                    yi = yi + y_pad + (y_max_edge - yi) * 0.2
                else:
                    va = "top"
                    yi -= y_pad

                maxes.Axes.text(self, xi, yi, s=label, va=va, ha=ha, **text_kw)

    def _pretransform_input_ipf(self, values):
        """Return unit vectors within the inverse pole figure from input
        data.

        A call to
        :meth:`orix.plot.StereographicPlot._pretransform_input` after
        this method is required to obtain cartesian coordinates to pass
        to Matplotlib's methods.

        Parameters
        ----------
        values : tuple of numpy.ndarray, Orientation, or Vector3d
            Spherical coordinates (azimuth, polar), orientations, or
            vectors. If spherical coordinates are given, they are
            assumed to describe unit vectors. Vectors will be made into
            unit vectors if they aren't already. If orientations are
            passed, the crystal directions returned are the sample
            direction rotated by the orientations.

        Returns
        -------
        v : Vector3d
            Crystal directions.
        """
        if len(values) == 2:  # (Azimuth, polar)
            v = Vector3d.from_polar(azimuth=values[0], polar=values[1])
        elif isinstance(values[0], Vector3d):
            v = values[0]
        else:  # Orientation
            v = values[0] * self._direction
        return v.in_fundamental_sector(self._symmetry)


mprojections.register_projection(InversePoleFigurePlot)


def _setup_inverse_pole_figure_plot(symmetry, direction=None, hemisphere=None):
    """Set up an inverse pole figure plot.

    Parameters
    ----------
    symmetry : ~orix.quaternion.Symmetry
        Laue group symmetry of crystal to plot directions with.
    direction : ~orix.vector.Vector3d, optional
        Sample direction to plot with respect to crystal directions. If
        not given, the out of plane direction, sample Z, is used.
    hemisphere : str, optional
        Which hemisphere(s) to plot the vectors in. If not given,
        "upper" is used. Options are "upper", "lower", and "both", which
        plots two projections side by side.

    Returns
    -------
    figure : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes
    """
    if direction is None:
        direction = Vector3d.zvector()

    n_plots = direction.size
    add_hemisphere_label = False
    if hemisphere is None:
        hemisphere = ["upper"] * n_plots
    elif hemisphere == "both":
        add_hemisphere_label = True
        hemisphere = ["upper"] * n_plots + ["lower"] * n_plots
        n_plots *= 2
        direction = Vector3d.stack((direction, direction))
    else:
        # Make iterable
        hemisphere = [hemisphere] * n_plots

    direction = direction.flatten()

    if n_plots <= 3:
        nrows = 1
        ncols = n_plots
    else:
        ncols = 3
        nrows = int(np.ceil(n_plots / 3))

    figure = plt.figure()
    axes = []
    subplot_kw = dict(projection="ipf", symmetry=symmetry)
    for i in range(n_plots):
        subplot_kw.update(dict(direction=direction[i], hemisphere=hemisphere[i]))
        ax = figure.add_subplot(nrows, ncols, i + 1, **subplot_kw)

        label_xy = np.column_stack(
            ax._projection.vector2xy(symmetry.fundamental_sector.vertices)
        )
        loc = None
        if label_xy.size != 0:
            # Expected title position
            expected_xy = np.array(
                [np.diff(ax.get_xlim())[0] / 2, np.max(ax.get_ylim())]
            )
            is_close = np.isclose(label_xy, expected_xy, atol=0.1).all(axis=1)
            if any(is_close) and plt.rcParams["axes.titley"] is None:
                loc = "left"

        ax.set_title(_get_ipf_title(direction[i]), loc=loc, fontweight="bold")

        if add_hemisphere_label:
            ax.show_hemisphere_label()

        axes.append(ax)

    return figure, np.asarray(axes)


def _get_ipf_title(direction):
    """Get a nicely formatted sample direction string from vector
    coordinates.

    Parameters
    ----------
    direction : ~orix.vector.Vector3d
        Single vector denoting the sample direction.

    Returns
    -------
    str
    """
    v = Vector3d(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    idx = np.where(np.isclose(direction.unit.dot(v).data, 1))[0]
    if idx.size != 0:
        return ["x", "y", "z"][idx[0]]
    else:
        return np.array_str(direction.data.squeeze()).strip("[]")


def _get_ipf_axes_labels(vertices, symmetry):
    r"""Get nicely formatted crystal direction strings from vector
    coordinates.

    Parameters
    ----------
    vertices : ~orix.vector.Vector3d
    symmetry : ~orix.quaternion.Symmetry
        Symmetry to determine which crystal directions `vertices`
        represent.

    Returns
    -------
    list of str
        List of strings, with -1 formatted like $\bar{1}$.
    """
    phase = Phase(point_group=symmetry)
    m = Miller(uvw=vertices.data, phase=phase)
    if symmetry.system in ["trigonal", "hexagonal"]:
        m.coordinate_format = "UVTW"
    axes = m.round(max_index=2).coordinates.astype(int)

    labels = []
    for ax in axes:
        label = r"[$"
        for i in ax:
            idx = str(abs(i))
            if i < 0:
                label += r"\bar{" + idx + r"}"
            else:
                label += idx
            label += " "
        label = label[:-1] + r"$]"
        labels.append(label)

    return labels
