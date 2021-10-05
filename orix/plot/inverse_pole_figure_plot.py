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

        vertices = fs.vertices
        if vertices.size > 0:
            # Add labels for crystal directions [uvw]/[UVTW]
            labels = _make_ipf_axes_labels(vertices, self._symmetry)
            x, y = self._projection.vector2xy(vertices)

            y_edge = self._edge_patch.get_path().vertices[:, 1]
            y_min_edge, y_max_edge = np.min(y_edge), np.max(y_edge)

            font_size = plt.rcParams["font.size"] + 4
            text_kw = dict(ha="center", va="center", fontsize=font_size)
            y_min, y_max = np.min(y), np.max(y)
            for label, xi, yi in zip(labels, x, y):
                # Determine x and y coordinates of label so that it
                # isn't placed over fundamental sector edge
                if np.isclose(yi, y_max) and y_min != y_max:
                    text_kw["va"] = "bottom"
                elif np.isclose(yi, y_min):
                    text_kw["va"] = "top"
                if y_min_edge < yi < y_max_edge:
                    yi += (y_max_edge - yi) / 2

                maxes.Axes.text(self, xi, yi, s=label, **text_kw)

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

        ax.set_title(_make_ipf_title(direction[i]), loc=loc, fontweight="bold")
        axes.append(ax)

    return figure, np.asarray(axes)


def _make_ipf_title(direction):
    v = Vector3d(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    idx = np.where(np.isclose(direction.dot(v).data, 1))[0]
    if idx.size != 0:
        return ["x", "y", "z"][idx[0]]
    else:
        return np.array_str(direction.data.squeeze()).strip("[]")


def _make_ipf_axes_labels(vertices, symmetry):
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
