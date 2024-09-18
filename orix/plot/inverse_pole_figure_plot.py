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

"""Inverse pole figure plot inheriting from
:class:`~orix.plot.StereographicPlot` for plotting of
:class:`~orix.vector.Vector3d`, typically parallel to sample directions,
rotated by orientations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.axes as maxes
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
import matplotlib.projections as mprojections
import matplotlib.pyplot as plt
import numpy as np

from orix.crystal_map import Phase
from orix.measure import pole_density_function
from orix.plot.direction_color_keys.direction_color_key_tsl import DirectionColorKeyTSL
from orix.plot.stereographic_plot import ZORDER, StereographicPlot
from orix.quaternion import Orientation
from orix.quaternion.symmetry import C1, Symmetry
from orix.vector import Miller, Vector3d


class InversePoleFigurePlot(StereographicPlot):
    """Inverse pole figure plot of :class:`~orix.vector.Vector3d`, which
    is a stereographic plot for showing sample directions with respect
    to a crystal reference frame.

    Parameters
    ----------
    *args
        Arguments passed to :class:`~orix.plot.StereographicPlot`.
    symmetry
        Laue group symmetry of crystal to plot directions with. If not
        given, point group C1 (only identity rotation) is used.
    direction
        Sample direction to plot with respect to crystal directions. If
        not given, the out of plane direction, sample Z, is used.
    hemisphere
        Which hemisphere(s) to plot the vectors in. If not given,
        ``"upper"`` is used. Options are ``"upper"``, ``"lower"`` and
        ``"both"``, which plots two projections side by side.
    **kwargs
        Keyword arguments passed to
        :class:`~orix.plot.StereographicPlot`.
    """

    name = "ipf"

    def __init__(
        self,
        *args: Any,
        symmetry: Optional[Symmetry] = None,
        direction: Optional[Vector3d] = None,
        hemisphere: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Create an inverse pole figure axis for plotting
        :class:`~orix.vector.Vector3d`.
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
    def _edge_patch(self) -> PathPatch:
        """Easy access to the fundamental sector border patch."""
        patches = self.patches
        return patches[self._has_collection(label="sa_sector", collections=patches)[1]]

    def pole_density_function(
        self,
        *args: Union[np.ndarray, Vector3d],
        resolution: float = 0.25,
        sigma: float = 5,
        log: bool = False,
        colorbar: bool = True,
        weights: Optional[np.ndarray] = None,
        **kwargs: Any,
    ):
        """Compute the Inverse Pole Density Function (IPDF) of vectors
        in the stereographic projection.

        The PDF is computed within the fundamental sector of the point
        group symmetry. See :cite:`rohrer2004distribution`.

        Parameters
        ----------
        *args
            Vector(s), or azimuth and polar angles of the vectors, the
            latter passed as two separate arguments.
        resolution
            The angular resolution of the sampling grid in degrees.
            Default value is 0.25.
        sigma
            The angular resolution of the applied broadening in degrees.
            Default value is 5.
        log
            If ``True`` the log(IPDF) is calculated. Default is
            ``True``.
        colorbar
            If ``True`` a colorbar is shown alongside the IPDF plot.
            Default is ``True``.
        weights
            The weights for the individual vectors. If not given, each
            vector is 1.
        **kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.pcolormesh`.

        See Also
        --------
        measure.pole_density_function
        StereographicPlot.pole_density_function
        Vector3d.pole_density_function
        """
        hist, (x, y) = pole_density_function(
            *args,
            resolution=resolution,
            sigma=sigma,
            log=log,
            hemisphere=self.hemisphere,
            symmetry=self._symmetry,
            weights=weights,
        )

        new_kwargs = dict(zorder=ZORDER["mesh"], clip_on=True)
        updated_kwargs = {**kwargs, **new_kwargs}

        # plot mesh
        updated_kwargs.setdefault("cmap", "magma")
        # mpl.QuadMesh handles masked values by default
        pc = self.pcolormesh(x, y, hist, **updated_kwargs)

        if colorbar:
            label = "MRD"
            if log:
                label = f"log({label})"
            plt.colorbar(pc, label=label, ax=self)

    def scatter(
        self,
        *args: Union[Tuple[np.ndarray, np.ndarray], Orientation, Vector3d],
        **kwargs: Any,
    ) -> None:
        """A scatter plot of sample directions rotated by orientations,
        or orientations to rotate sample directions with.

        Parameters
        ----------
        *args
            Spherical coordinates (azimuth, polar), orientations, or
            vectors. If spherical coordinates are given, they are
            assumed to describe unit vectors. Vectors will be made into
            unit vectors if they aren't already. If orientations are
            passed, the crystal directions returned are the sample
            :attr:`direction` rotated by the orientations.
        **kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.scatter`.

        See Also
        --------
        matplotlib.axes.Axes.scatter
        """
        crystal_directions = self._pretransform_input_ipf(args)
        super().scatter(crystal_directions, **kwargs)

    def show_hemisphere_label(self, **kwargs: Any) -> None:
        """Add a hemisphere label (``"upper"``/``"lower"``) to the upper
        left outside the inverse pole figure.

        Parameters
        ----------
        **kwargs
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

    def _add_crystal_direction_labels(self) -> None:
        """Add appropriately placed and nicely formatted crystal
        direction labels [uvw] or [UVTW] to the sector corners.
        """
        fs = self._symmetry.fundamental_sector
        vertices = fs.vertices
        if vertices.size > 0:
            center = fs.center.y[0]

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

    def _pretransform_input_ipf(
        self, values: Union[Tuple[np.ndarray, np.ndarray], Orientation, Vector3d]
    ) -> Vector3d:
        """Return unit vectors within the inverse pole figure from input
        data.

        A call to
        :meth:`orix.plot.StereographicPlot._pretransform_input` after
        this method is required to obtain cartesian coordinates to pass
        to Matplotlib's methods.

        Parameters
        ----------
        values
            Spherical coordinates (azimuth, polar), orientations, or
            vectors. If spherical coordinates are given, they are
            assumed to describe unit vectors. Vectors will be made into
            unit vectors if they aren't already. If orientations are
            passed, the crystal directions returned are the sample
            direction rotated by the orientations.

        Returns
        -------
        v
            Crystal directions.
        """
        if len(values) == 2:  # (Azimuth, polar)
            v = Vector3d.from_polar(azimuth=values[0], polar=values[1])
        elif isinstance(values[0], Vector3d):
            v = values[0]
        else:  # Orientation
            v = values[0] * self._direction
        return v.in_fundamental_sector(self._symmetry)

    def plot_ipf_color_key(self, show_title: bool = True) -> None:
        """Plot an IPF color key code on this axis.

        Parameters
        ----------
        show_title
            Whether to display the Laue group name as the axes title.
            Default is ``True``.

        Notes
        -----
        This function may be used to plot the IPF color key alongside
        another plot where the same key was used to color
        :class:`~orix.quaternion.Orientation` or
        :class:`~orix.vector.Vector3d`.
        """
        symmetry = self._symmetry
        direction_color_key = DirectionColorKeyTSL(symmetry)

        rgba_grid, (x_lim, y_lim) = direction_color_key._create_rgba_grid(
            return_extent=True
        )

        if show_title:
            label_xy = np.column_stack(
                self._projection.vector2xy(symmetry.fundamental_sector.vertices)
            )
            loc = None
            if label_xy.size != 0:
                # Expected title position
                expected_xy = np.array(
                    [np.diff(self.get_xlim())[0] / 2, np.max(self.get_ylim())]
                )
                is_close = np.isclose(label_xy, expected_xy, atol=0.1).all(axis=1)
                if any(is_close) and plt.rcParams["axes.titley"] is None:
                    loc = "left"
            self.set_title(symmetry.name, loc=loc, fontweight="bold")

        self.stereographic_grid(False)
        self._edge_patch.set_linewidth(1.5)
        self.imshow(rgba_grid, extent=x_lim + y_lim, zorder=0)


mprojections.register_projection(InversePoleFigurePlot)


def _setup_inverse_pole_figure_plot(
    symmetry: Symmetry,
    direction: Optional[Vector3d] = None,
    hemisphere: Optional[str] = None,
    figure_kwargs: Optional[Dict] = None,
) -> Tuple[Figure, np.ndarray]:
    """Set up an inverse pole figure plot.

    Parameters
    ----------
    symmetry
        Laue group symmetry of crystal to plot directions with.
    direction
        Sample direction to plot with respect to crystal directions. If
        not given, the out of plane direction, sample Z, is used.
    hemisphere
        Which hemisphere(s) to plot the vectors in. If not given,
        ``"upper"`` is used. Options are ``"upper"``, ``"lower"``, and
        ``"both"``, which plots two projections side by side.
    figure_kwargs
        Dictionary of keyword arguments passed to
        :func:`matplotlib.pyplot.figure`.

    Returns
    -------
    figure
    axes
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

    if figure_kwargs is None:
        figure_kwargs = {"layout": "tight"}
    figure = plt.figure(**figure_kwargs)
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


def _get_ipf_title(direction: Vector3d) -> str:
    """Get a nicely formatted sample direction string from vector
    coordinates.

    Parameters
    ----------
    direction
        Single vector denoting the sample direction.

    Returns
    -------
    str
    """
    v = Vector3d(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    idx = np.where(np.isclose(direction.unit.dot(v), 1))[0]
    if idx.size != 0:
        return ["x", "y", "z"][idx[0]]
    else:
        return np.array_str(direction.data.squeeze()).strip("[]")


def _get_ipf_axes_labels(vertices: Vector3d, symmetry: Symmetry) -> List[str]:
    r"""Get nicely formatted crystal direction strings from vector
    coordinates.

    Parameters
    ----------
    vertices
    symmetry
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
