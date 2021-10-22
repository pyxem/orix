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

"""Stereographic plot inheriting from :class:`~matplotlib.axes.Axes` for
plotting :class:`~orix.vector.Vector3d`.
"""

from copy import deepcopy

from matplotlib import rcParams
import matplotlib.axes as maxes
import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.projections as mprojections
import numpy as np

from orix.plot._symmetry_marker import (
    TwoFoldMarker,
    ThreeFoldMarker,
    FourFoldMarker,
    SixFoldMarker,
)
from orix.projections import InverseStereographicProjection, StereographicProjection
from orix.vector import Vector3d
from orix.vector.fundamental_sector import _closed_edges_in_hemisphere


ZORDER = dict(text=6, scatter=5, symmetry_marker=4, draw_circle=3)


class StereographicPlot(maxes.Axes):
    """Stereographic plot for plotting :class:`~orix.vector.Vector3d`.

    Inherits from :class:`~matplotlib.axes.Axes`.
    """

    name = "stereographic"
    _pad_xy = 0.05

    def __init__(
        self,
        *args,
        hemisphere="upper",
        azimuth_resolution=10,
        polar_resolution=10,
        **kwargs,
    ):
        """Create an axis for plotting :class:`~orix.vector.Vector3d`.

        Parameters
        ----------
        args
            Arguments passed to :meth:`matplotlib.axes.Axes.__init__`.
        hemisphere : str, optional
            Which hemisphere to plot vectors in, either "upper"
            (default) or "lower".
        azimuth_resolution : int or float, optional
            Resolution of azimuth grid lines in degrees. Default is 10
            degrees.
        polar_resolution : int or float, optional
            Resolution of polar grid lines in degrees. Default is 10
            degrees.
        kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.__init__`.
        """
        self.hemisphere = hemisphere
        self._azimuth_resolution = azimuth_resolution
        self._polar_resolution = polar_resolution

        # Custom attribute to keep track of whether grid is on or off
        self._stereographic_grid = None

        super().__init__(*args, **kwargs)

        # Set ratio of y-unit to x-unit by adjusting the physical
        # dimension of the Axes (box), and centering the anchor (C)
        self.set_aspect("equal", adjustable="box", anchor="C")
        self.clear()

    def clear(self):
        super().clear()

        self.xaxis.set_ticks_position("none")
        self.yaxis.set_ticks_position("none")
        self.xaxis.set_tick_params(label1On=False)
        self.yaxis.set_tick_params(label1On=False)

        self.set_xlim(-1 - self._pad_xy, 1 + self._pad_xy)
        self.set_ylim(-1 - self._pad_xy, 1 + self._pad_xy)

        spines = self.spines
        for spine in spines.values():
            spine.set_visible(False)

        self.add_patch(
            mpatches.Circle(
                xy=(0, 0),
                radius=1,
                facecolor="none",
                edgecolor="k",
                label="sa_circle",
            )
        )

        # Don't show rectangular grid
        self.grid(False)
        self.stereographic_grid(rcParams["axes.grid"])

    def format_coord(self, x, y):
        if np.sqrt(np.sum(np.square([x, y]))) > 1:
            return ""
        else:
            azimuth, polar = self._inverse_projection.xy2spherical(x, y)
            azimuth = azimuth[0]
            polar = polar[0]
            azimuth_deg = np.rad2deg(azimuth)
            polar_deg = np.rad2deg(polar)
            return (
                "\N{GREEK SMALL LETTER PHI}={:.2f}\N{GREEK SMALL LETTER PI} "
                "({:.2f}\N{DEGREE SIGN}), "
                "\N{GREEK SMALL LETTER theta}={:.2f}\N{GREEK SMALL LETTER PI} "
                "({:.2f}\N{DEGREE SIGN})"
            ).format(azimuth / np.pi, azimuth_deg, polar / np.pi, polar_deg)

    def plot(self, *args, **kwargs):
        """Draw straight lines between vectors.

        This method overwrites :meth:`matplotlib.axes.Axes.plot`, see
        that method's docstring for parameters.

        Parameters
        ----------
        args : Vector3d or tuple of float or numpy.ndarray
            Vector(s), or azimuth and polar angles, the latter two
            passed as separate arguments (not keyword arguments).
        kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.plot`.

        See Also
        --------
        matplotlib.axes.Axes.plot
        """
        new_kwargs = dict(clip_on=True, linewidth=2, color="k", linestyle="-")
        x, y, _, updated_kwargs = self._prepare_to_call_inherited_method(
            args, kwargs, new_kwargs, sort=True
        )
        if x.size == 0:
            return

        super().plot(x, y, **updated_kwargs)

    def scatter(self, *args, **kwargs):
        """A scatter plot of vectors.

        This method overwrites :meth:`matplotlib.axes.Axes.scatter`, see
        that method's docstring for parameters.

        Parameters
        ----------
        args : Vector3d or tuple of float or numpy.ndarray
            Vector(s), or azimuth and polar angles, the latter two
            passed as separate arguments (not keyword arguments).
        kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.scatter`.

        See Also
        --------
        matplotlib.axes.Axes.scatter
        """
        new_kwargs = dict(zorder=ZORDER["scatter"], clip_on=False)
        out = self._prepare_to_call_inherited_method(args, kwargs, new_kwargs)
        x, y, visible, updated_kwargs = out
        if x.size == 0:
            return

        # Color(s) and size(s)
        c = updated_kwargs.pop("c", "C0")
        c = _get_array_of_values(value=c, visible=visible)
        s = updated_kwargs.pop("s", None)
        if s is not None:
            s = _get_array_of_values(value=s, visible=visible)

        super().scatter(x, y, c=c, s=s, **updated_kwargs)

    def text(self, *args, **kwargs):
        """Add text to the axes.

        This method overwrites :meth:`matplotlib.axes.Axes.text`, see
        that method's docstring for parameters.

        Parameters
        ----------
        args : Vector3d or tuple of float or numpy.ndarray
            Vector(s), or azimuth and polar angles, the latter two
            passed as separate arguments (not keyword arguments).
        kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.text`.

        See Also
        --------
        matplotlib.axes.Axes.text
        """
        new_kwargs = dict(va="bottom", ha="center", zorder=ZORDER["text"])
        out = self._prepare_to_call_inherited_method(args, kwargs, new_kwargs)
        x, y, _, updated_kwargs = out
        if x.size == 0:
            return
        super().text(x, y, **updated_kwargs)

    # ----------- Custom attributes and methods below here ----------- #

    @property
    def hemisphere(self):
        """Hemisphere to plot.

        Returns
        -------
        str
            "upper" or "lower" plots the upper or lower hemisphere
            vectors. :attr:`pole` is derived from this attribute.
        """
        return self._hemisphere

    @hemisphere.setter
    def hemisphere(self, value):
        """Set hemisphere to plot."""
        value = value.lower()
        if value in ["upper", "lower"]:
            self._hemisphere = value
        else:
            raise ValueError(f"Hemisphere must be 'upper' or 'lower', not {value}")

    @property
    def pole(self):
        """Projection pole, either -1 or 1, where -1 (1) means the
        projection point of the stereographic transform is the south
        (north) pole [00-1] ([001]), i.e. only vectors with z > 0 (z <
        0) are plotted.

        Derived from :attr:`hemisphere`.
        """
        return {"upper": -1, "lower": 1}[self.hemisphere]

    @property
    def _projection(self):
        return StereographicProjection(self.pole)

    @property
    def _inverse_projection(self):
        return InverseStereographicProjection(self.pole)

    def draw_circle(self, *args, opening_angle=np.pi / 2, steps=100, **kwargs):
        r"""Draw great or small circles with a given `opening_angle` to
        one or multiple vectors.

        A vector must be present in the current hemisphere for its
        circle to be drawn.

        Parameters
        ----------
        args : Vector3d or tuple of float or numpy.ndarray
            Vector(s), or azimuth and polar angles defining vectors, the
            latter two passed as separate arguments (not keyword
            arguments). Circles are drawn perpendicular to these with a
            given `opening_angle`.
        opening_angle : float or numpy.ndarray, optional
            Opening angle(s) around the vector(s). Default is
            :math:`\pi/2`, giving a great circle. If an array is passed,
            its size must be equal to the number of circles to draw.
        steps : int, optional
            Number of vectors to describe each circle, default is 100.
        kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.plot` to alter the circles'
            appearance.

        See Also
        --------
        orix.vector.Vector3d.get_circle
        """
        out = self._prepare_to_call_inherited_method(args, kwargs)
        x, y, visible, updated_kwargs = out
        if x.size == 0:
            return

        # Get set of `steps` vectors delineating a circle per vector
        v = self._inverse_projection.xy2vector(x, y)
        circles = v.get_circle(opening_angle=opening_angle, steps=steps).unit

        # Enable using one color per circle
        color = kwargs.pop("color", "C0")
        color2 = _get_array_of_values(value=color, visible=visible)

        # Set above which elements circles will appear (zorder)
        new_kwargs = dict(zorder=ZORDER["draw_circle"], clip_on=True)
        for k, v in new_kwargs.items():
            kwargs.setdefault(k, v)

        for i, c in enumerate(circles):
            self.plot(c.azimuth.data, c.polar.data, color=color2[i], **kwargs)

    def restrict_to_sector(self, sector):
        """Restrict the stereographic axis to a
        :class:`~orix.vector.FundamentalSector`, typically obtained from
        :attr:`~orix.quaternion.Symmetry.fundamental_sector`.

        Parameters
        ----------
        sector : ~orix.vector.FundamentalSector
            Fundamental sector with edges delineating a fundamental
            sector.
        """
        original_pole = deepcopy(sector._pole)
        sector._pole = self.pole
        edges = sector.edges
        if edges.size == 0:
            return
        edges = _closed_edges_in_hemisphere(edges, sector, pole=self.pole)
        sector._pole = original_pole
        if edges.size == 0:
            return
        x, y, _ = self._pretransform_input((edges,))

        pad = 0.01
        self.set_xlim(np.min(x) - pad, np.max(x) + pad)
        self.set_ylim(np.min(y) - pad, np.max(y) + pad)
        self.margins(0, 0)
        self.patches[0].set_visible(False)
        patch = mpatches.PathPatch(
            mpath.Path(np.column_stack([x, y]), closed=True),
            facecolor="none",
            edgecolor="k",
            linewidth=1,
            label="sa_sector",
        )
        self.add_patch(patch)
        self.set_clip_path(patch)
        labels = ["sa_azimuth_grid", "sa_polar_grid"]
        for c in self.collections:
            if c.get_label() in labels:
                c.set_clip_path(patch)

    def show_hemisphere_label(self, **kwargs):
        """Add a hemisphere label ("upper"/"lower") to the upper left of
        the plane.

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
        super().text(-0.71, 0.71, s=self.hemisphere, **new_kwargs)

    def set_labels(self, xlabel="x", ylabel="y", zlabel="z", **kwargs):
        """Set the reference frame's axes labels.

        Parameters
        ----------
        xlabel : str, False or None, optional
            X axis label, default is "x". If False or None, this label
            is not shown.
        ylabel : str, False or None, optional
            Y axis label, default is "y". If False or None, this label
            is not shown.
        zlabel : str, False or None, optional
            Z axis label, default is "z". If False or None, this label
            is not shown.
        """
        pos = [(1, 0), (0, 1), (0, 0)]
        for (x, y), label in zip(pos, [xlabel, ylabel, zlabel]):
            if label not in [None, False]:
                self._set_label(x=x, y=y, label=label, **kwargs)

    def stereographic_grid(
        self, show_grid=None, azimuth_resolution=None, polar_resolution=None
    ):
        """Turn a stereographic grid on or off, and set the azimuth and
        polar grid resolution in degrees.

        Parameters
        ----------
        show_grid : bool, optional
            Whether to show grid lines. If any keyword arguments are
            passed, this is set to True. If not given and there are no
            keyword arguments, the grid lines are toggled.
        azimuth_resolution : float, optional
            Azimuth grid resolution in degrees. Default is 10 degrees.
            This can also be set upon initialization of the axes by
            passing `azimuth_resolution` to `subplot_kw`.
        polar_resolution : float, optional
            Polar grid resolution in degrees. Default is 10 degrees.
            This can also be set upon initialization of the axes by
            passing `polar_resolution` to `subplot_kw`.

        See Also
        --------
        matplotlib.axes.Axes.grid
        """
        if (
            show_grid is None
            and self._stereographic_grid in [None, False]
            or show_grid is None
            and (azimuth_resolution is not None or polar_resolution is not None)
            or show_grid is True
        ) and hasattr(self, "patch"):
            self._azimuth_grid(azimuth_resolution)
            self._polar_grid(polar_resolution)
            self._stereographic_grid = True
        elif show_grid in [None, False] and self._stereographic_grid is True:
            # Remove grid
            has_azimuth, index_azimuth = self._has_collection(
                "sa_azimuth_grid", self.collections
            )
            has_polar, index_polar = self._has_collection(
                "sa_polar_grid", self.collections
            )
            if has_azimuth:
                if index_polar > index_azimuth:
                    index_polar -= 1
                self.collections[index_azimuth].remove()
            if has_polar:
                self.collections[index_polar].remove()
            self._stereographic_grid = False

    def symmetry_marker(self, v, fold, **kwargs):
        """Plot 2-, 3- 4- or 6-fold symmetry marker(s).

        Parameters
        ----------
        v : Vector3d
            Position of the marker(s) to plot.
        fold : int
            Which symmetry element to plot, can be either 2, 3, 4 or 6.
        kwargs
            Keyword arguments passed to :meth:`scatter`.
        """
        if fold not in [2, 3, 4, 6]:
            raise ValueError("Can only plot 2-, 3-, 4- or 6-fold elements.")

        marker_classes = {
            "2": TwoFoldMarker,
            "3": ThreeFoldMarker,
            "4": FourFoldMarker,
            "6": SixFoldMarker,
        }
        marker = marker_classes[str(fold)](v, size=kwargs.pop("s", 1))

        new_kwargs = dict(zorder=ZORDER["symmetry_marker"], clip_on=False)
        for k, v in new_kwargs.items():
            kwargs.setdefault(k, v)

        for vec, marker, marker_size in marker:
            self.scatter(vec, marker=marker, s=marker_size, **kwargs)

        # TODO: Find a way to control padding, so that markers aren't
        #  clipped

    def _azimuth_grid(self, resolution=None):
        """Set the azimuth grid resolution in degrees.

        Parameters
        ----------
        resolution : float, optional
            Azimuth grid resolution in degrees. Default is 10 degrees.
            This can also be set upon initialization of the axes by
            passing `azimuth_resolution` to `subplot_kw`.

        See Also
        --------
        polar_grid
        matplotlib.axes.Axes.grid
        """
        if resolution is not None:
            self._azimuth_resolution = resolution

        azimuth_start = np.arange(0, np.pi, np.radians(self._azimuth_resolution))
        polar = np.full(azimuth_start.size, np.pi / 2)
        if self.hemisphere == "lower":
            polar += 1e-9
        v_start = Vector3d.from_polar(azimuth_start, polar)
        x_start, y_start = self._projection.vector2xy(v_start)
        v_end = Vector3d.from_polar(azimuth_start + np.pi, polar)
        x_end, y_end = self._projection.vector2xy(v_end)

        kwargs = dict(
            linewidths=rcParams["grid.linewidth"],
            linestyle=rcParams["grid.linestyle"],
            alpha=rcParams["grid.alpha"],
            color=rcParams["grid.color"],
            antialiased=True,
        )

        label = "sa_azimuth_grid"
        lines = np.stack(((x_start, x_end), (y_start, y_end))).T
        lines_collection = mcollections.LineCollection(lines, label=label, **kwargs)
        has_collection, index = self._has_collection(label, self.collections)
        if has_collection:
            self.collections[index].remove()
        has_sector, sector_index = self._has_collection("sa_sector", self.patches)
        if has_sector:
            lines_collection.set_clip_path(self.patches[sector_index])
        self.add_collection(lines_collection)

    @staticmethod
    def _has_collection(label, collections):
        labels = [c.get_label() for c in collections]
        for i in range(len(labels)):
            if label == labels[i]:
                return True, i
        return False, -1

    def _polar_grid(self, resolution=None):
        """Set the polar grid resolution in degrees.

        Parameters
        ----------
        resolution : float, optional
            Polar grid resolution in degrees. Default is 15 degrees.
            This can also be set upon initialization of the axes by
            passing `polar_resolution` to the `subplot_kw` dictionary.

        See Also
        --------
        azimuth_grid
        matplotlib.axes.Axes.grid
        """
        if resolution is not None:
            self._polar_resolution = resolution

        res = np.radians(self._polar_resolution)

        polar = np.arange(res, np.pi, res)
        v = Vector3d.from_polar(np.zeros(polar.size), polar)
        radii, _ = self._projection.vector2xy(v)

        ec = rcParams["grid.color"]
        kwargs = dict(
            xy=(0, 0),
            linewidth=rcParams["grid.linewidth"],
            linestyle=rcParams["grid.linestyle"],
            alpha=rcParams["grid.alpha"],
            ec=ec,
            fc="none",
            antialiased=True,
        )

        circles = []
        for r in radii:
            circles.append(mpatches.Circle(radius=r, **kwargs))
        label = "sa_polar_grid"
        circles_collection = mcollections.PatchCollection(
            circles,
            label=label,
            edgecolors=kwargs["ec"],
            facecolors=kwargs["fc"],
        )
        has_collection, index = self._has_collection(label, self.collections)
        if has_collection:
            self.collections[index].remove()
        has_sector, sector_index = self._has_collection("sa_sector", self.patches)
        if has_sector:
            circles_collection.set_clip_path(self.patches[sector_index])
        self.add_collection(circles_collection)

    def _prepare_to_call_inherited_method(
        self, args, kwargs, new_kwargs=None, sort=False
    ):
        """Prepare arguments and keyword arguments passed to methods in
        :class:`StereographicPlot` inherited from
        :class:`matplotlib.axes.Axes`.

        Parameters
        ----------
        args
            Any arguments passed to the :class:`StereographicPlot`
            method.
        kwargs : dict
            Any arguments passed to the :class:`StereographicPlot`
            method.
        new_kwargs : dict
            Any default keyword arguments to be passed to the inherited
            method.

        Returns
        -------
        x : numpy.ndarray
        y : numpy.ndarray
        visible : numpy.ndarray
        updated_kwargs : dict
        """
        updated_kwargs = kwargs
        if new_kwargs is not None:
            for k, v in new_kwargs.items():
                updated_kwargs.setdefault(k, v)
        x, y, visible = self._pretransform_input(args, sort=sort)
        return x, y, visible, updated_kwargs

    def _pretransform_input(self, values, sort=False):
        """Return arrays of (x, y) from input data.

        Parameters
        ----------
        values : tuple of numpy.ndarray or Vector3d
            Spherical coordinates (azimuth, polar) or vectors. If
            spherical coordinates are given, they are assumed to
            describe unit vectors. Vectors will be made into unit
            vectors if they aren't already.

        Returns
        -------
        x : numpy.ndarray
            Stereographic x coordinates of unit vectors.
        y : numpy.ndarray
            Stereographic y coordinates of unit vectors.
        """
        pole = self.pole
        if len(values) == 2:
            azimuth, polar = values[0], values[1]
            if sort:
                order = _order_in_hemisphere(polar, pole)
                azimuth = azimuth[order]
                polar = polar[order]
            x, y = self._projection.spherical2xy(azimuth=azimuth, polar=polar)
            v = self._inverse_projection.xy2vector(x, y)
        else:
            try:
                v = values[0].flatten().unit
                if sort:
                    order = _order_in_hemisphere(v.polar.data, pole)
                    v = v[order]
                x, y = self._projection.vector2xy(v)
            except (ValueError, AttributeError):
                raise ValueError(
                    "Accepts only one (Vector3d) or two (azimuth, polar) input "
                    "arguments"
                )
        visible = v <= self._projection.region
        return x, y, visible

    def _set_label(self, x, y, label, **kwargs):
        bbox_dict = dict(boxstyle="round, pad=0.1", fc="w", ec="w")
        new_kwargs = dict(ha="center", va="center", bbox=bbox_dict)
        new_kwargs.update(kwargs)
        super().text(x=x, y=y, s=label, **new_kwargs)


mprojections.register_projection(StereographicPlot)


def _get_array_of_values(value, visible):
    """Return a usable array of `value` with the correct size
    even though `value` doesn't have as many elements as `visible.size`,
    to be iterated over along with `True` elements in `visible`.

    Parameters
    ----------
    value : str, float, or a list of str or float
        Typically a keyword argument value to be passed to some
        Matplotlib routine.
    visible : numpy.ndarray
        Boolean array with as many elements as input vectors, only some
        of which are visible in the hemisphere (`True`).

    Returns
    -------
    numpy.ndarray
        An array populated with `value` of a size equal to the number of
        True elements in `visible`.
    """
    n = visible.size
    if not isinstance(value, str) and hasattr(value, "__iter__") and len(value) != n:
        value = value[0]
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        value = [value] * n
    return np.asarray(value)[visible]


def _is_visible(polar, pole):
    """Return a boolean array describing whether the vector which the
    polar angles belong to are visible in the current hemisphere.

    Parameters
    ----------
    polar : numpy.ndarray
    pole : int

    Returns
    -------
    numpy.ndarray
        Boolean array with True for polar angles corresponding to
        vectors visible in this hemisphere.
    """
    if pole == -1:
        return polar <= np.pi / 2
    else:  # pole == 1
        return polar >= np.pi / 2


def _order_in_hemisphere(polar, pole):
    """Return order of vectors based on polar angles, so that the ones
    corresponding to vectors visible in this hemisphere are shifted to
    the start of the arrays.

    Used in :meth:`StereographicPlot._pretransform_input` when
    `sort=True`.

    Parameters
    ----------
    polar : numpy.ndarray
    pole : int

    Returns
    -------
    numpy.ndarray or None
        If no vectors are visible, None is returned.
    """
    visible = _is_visible(polar, pole)
    if visible.size == 0 or not np.any(visible):
        return

    indices = np.asarray(visible != visible[0]).nonzero()[0]
    order = np.arange(visible.size)
    if indices.size != 0:
        order = np.roll(order, shift=-(indices[-1] + 1))

    return order
