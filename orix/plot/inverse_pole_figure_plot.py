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
    name = "ipf"

    def __init__(
        self,
        *args,
        symmetry=None,
        direction=None,
        hemisphere=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if hemisphere is not None:
            self.hemisphere = hemisphere

        if direction is None:
            direction = Vector3d.zvector()
        self._direction = direction

        if symmetry is None:
            symmetry = C1
        self._symmetry = symmetry

        fs = self._symmetry.fundamental_sector
        self.restrict_to_sector(fs)

        self._add_crystal_direction_labels()

    @property
    def _edge_patch(self):
        patches = self.patches
        return patches[self._has_collection(label="sa_sector", collections=patches)[1]]

    def scatter(self, *args, **kwargs):
        vc = self._pretransform_input_ipf(args)
        super().scatter(vc, **kwargs)

    def show_hemisphere_label(self, **kwargs):
        """Add a hemisphere label ("upper"/"lower") to the upper left
        outside the inverse pole figure.

        Parameters
        ----------
        kwargs
            Keyword arguments passed to
            :func:`matplotlib.axes.Axes.text`.

        See Also
        --------
        hemisphere
        """
        new_kwargs = dict(ha="right", va="bottom")
        new_kwargs.update(kwargs)
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
        if len(values) == 2:  # (Azimuth, polar)
            v = Vector3d.from_polar(azimuth=values[0], polar=values[1])
        elif isinstance(values[0], Vector3d):
            v = values[0]
        else:  # Orientation
            v = values[0] * self._direction
        return v.in_fundamental_sector(self._symmetry)


mprojections.register_projection(InversePoleFigurePlot)


def _setup_inverse_pole_figure_plot(symmetry, direction=None, hemisphere=None):
    if direction is None:
        direction = Vector3d.zvector()

    n_plots = direction.size
    if hemisphere is None:
        hemisphere = ["upper"] * n_plots
    elif hemisphere == "both":
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
    for i in range(n_plots):
        subplot_kw = dict(
            projection="ipf",
            symmetry=symmetry,
            direction=direction[i],
            hemisphere=hemisphere[i],
        )
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
    idx = np.where(np.isclose(direction.dot(v).data, 1))[0]
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
