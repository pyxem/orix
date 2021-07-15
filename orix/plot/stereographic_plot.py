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

from collections import OrderedDict

import matplotlib.transforms as mtransforms
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.axis import XAxis, YAxis
from matplotlib.patches import Wedge
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
import numpy as np

from orix.projections import InverseStereographicProjection, StereographicProjection
from orix.plot._symmetry_marker import (
    TwoFoldMarker,
    ThreeFoldMarker,
    FourFoldMarker,
    SixFoldMarker,
)
from orix.vector import Vector3d


# Inspiration and resources for the stereographic plot:
# https://matplotlib.org/stable/devel/add_new_projection.html
# https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
# https://matplotlib.org/stable/gallery/misc/custom_projection.html
# https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/projections/polar.py


class StereographicTransform(mtransforms.Transform):
    """The stereographic transform."""

    input_dims = output_dims = 2

    def __init__(self, pole=-1):
        """Create a new stereographic transform.

        Parameters
        ----------
        pole : int, optional
            -1 or 1, where -1 (1) means the projection point of the
            stereographic transform is the south (north) pole [00-1]
            ([001]), i.e. only vectors with z > 0 (z < 0) are returned.
        """
        super().__init__()
        self.pole = pole

    def transform_non_affine(self, values):
        # (azimuth, polar) to (X, Y)
        azimuth, polar = values.T
        sp = StereographicProjection(pole=self.pole)
        x, y = sp.spherical2xy(azimuth=azimuth, polar=polar)
        return np.column_stack([x, y])

    def transform_path_non_affine(self, path):
        ipath = path.interpolated(path._interpolation_steps)
        return Path(self.transform(ipath.vertices), ipath.codes)

    def inverted(self):
        return InvertedStereographicTransform(pole=self.pole)


class InvertedStereographicTransform(mtransforms.Transform):
    input_dims = output_dims = 2

    def __init__(self, pole=-1):
        super().__init__()
        self.pole = pole

    def transform_non_affine(self, values):
        # (X, Y) to (azimuth, polar)
        x, y = values.T
        isp = InverseStereographicProjection(pole=self.pole)
        azimuth, polar = isp.xy2spherical(x=x, y=y)
        return np.column_stack([azimuth, polar])

    def inverted(self):
        return StereographicTransform(pole=self.pole)


class StereographicAffine(mtransforms.Affine2DBase):
    """The affine part of the stereographic projection. Scales the
    output so that maximum polar angle rests on the edge of the axes
    circle.
    """

    def __init__(self, limits, pole=-1):
        """`limits` is the view limit of the data. The only part of its
        bounds that is used is the y limits (for the polar limits).
        The azimuth range is handled by the non-affine transform.
        """
        super().__init__()
        self.pole = pole
        self._limits = limits
        self.set_children(limits)
        self._mtx = None

    __str__ = mtransforms._make_str_method("_limits", "pole")

    def get_matrix(self):
        if self._invalid:
            polar_max = self._limits.ymax
            st = StereographicTransform(pole=self.pole)
            y_scale, _ = st.transform((0, polar_max))
            self._mtx = mtransforms.Affine2D().scale(0.5 / y_scale).translate(0.5, 0.5)
            self._inverted = None
            self._invalid = 0
        return self._mtx


ZORDER = dict(text=6, scatter=100, symmetry_marker=4, draw_circle=3)


class StereographicPlot(Axes):
    """Stereographic projection plot. The projection is an equal angle
    projection and is typically used for visualizing 3D vectors in a 2D
    plane.

    https://en.wikipedia.org/wiki/Stereographic_projection

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from orix import plot, vector
    >>> fig, ax = plt.subplots(subplot_kw=dict(projections="stereographic"))
    >>> ax.scatter(vector.Vector3d([[0, 0, 1], [1, 0, 1]]))
    """

    # Axes: x - azimuth, y - polar

    name = "stereographic"
    _hemisphere = "upper"

    def __init__(self, *args, azimuth_resolution=10, polar_resolution=10, **kwargs):
        self._azimuth_cap = 2 * np.pi
        self._azimuth_resolution = azimuth_resolution
        self._polar_cap = np.pi / 2
        self._polar_resolution = polar_resolution

        # Custom attribute to keep track of whether grid is on or off
        self._stereographic_grid = None

        super().__init__(*args, **kwargs)
        self.use_sticky_edges = True  # ??
        # Set ratio of y-unit to x-unit by adjusting the physical
        # dimension of the Axes (box), and centering the anchor (C)
        self.set_aspect("equal", adjustable="box", anchor="C")
        self.clear()

    def _init_axis(self):
        # Need to override these to get rid of spines
        self.xaxis = XAxis(self)
        self.yaxis = YAxis(self)
        self._update_transScale()

    def clear(self):
        super().clear()

        self.xaxis.set_ticks_position("none")
        self.yaxis.set_ticks_position("none")
        self.xaxis.set_tick_params(label1On=False)
        self.yaxis.set_tick_params(label1On=False)

        self.title.set_y(1.05)

        self.spines["start"].set_visible(False)
        self.spines["end"].set_visible(False)
        self.set_xlim(0, self._azimuth_cap)
        self.set_ylim(0, self._polar_cap)

        self.polar_grid()
        self.azimuth_grid()
        self.grid(rcParams["axes.grid"])

    def _set_lim_and_transforms(self):
        self._originViewLim = mtransforms.LockableBbox(self.viewLim)
        self.transScale = mtransforms.TransformWrapper(mtransforms.IdentityTransform())
        self.axesLim = _WedgeBbox(
            center=(0.5, 0.5), viewLim=self.viewLim, originLim=self._originViewLim
        )

        # Scale the wedge to fill the axes
        self.transWedge = mtransforms.BboxTransformFrom(self.axesLim)

        # Data (azimuth, polar) space into rectilinear space (X, Y)
        self.transProjection = StereographicTransform(pole=self.pole)
        # Rectilinear space (X, Y) into axes space (0, 0) to (1, 1)
        self.transProjectionAffine = StereographicAffine(
            limits=self.viewLim, pole=self.pole
        )
        # Axes space to display space. Scale the axes to fill the figure
        self.transAxes = mtransforms.BboxTransformTo(self.bbox)

        # Data -> display coordinates
        self.transData = (
            self.transScale
            + self.transProjection
            + self.transProjectionAffine
            + self.transWedge
            + self.transAxes
        )

        self._xaxis_transform = (
            mtransforms.blended_transform_factory(
                x_transform=mtransforms.IdentityTransform(),
                y_transform=mtransforms.BboxTransformTo(self.viewLim),
            )
            + self.transData
        )
        self._xaxis_text_transform = self.transData

        self._yaxis_transform = (
            mtransforms.blended_transform_factory(
                x_transform=mtransforms.BboxTransformTo(self.viewLim),
                y_transform=mtransforms.IdentityTransform(),
            )
            + self.transData
        )
        self._yaxis_text_transform = mtransforms.TransformWrapper(self.transData)

    @staticmethod
    def format_coord(azimuth, polar):
        azimuth_deg = np.rad2deg(azimuth)
        polar_deg = np.rad2deg(polar)
        return (
            "\N{GREEK SMALL LETTER PHI}={:.2f}\N{GREEK SMALL LETTER PI} "
            "({:.2f}\N{DEGREE SIGN}), "
            "\N{GREEK SMALL LETTER theta}={:.2f}\N{GREEK SMALL LETTER PI} "
            "({:.2f}\N{DEGREE SIGN})"
        ).format(azimuth / np.pi, azimuth_deg, polar / np.pi, polar_deg)

    def get_xaxis_transform(self, which="grid"):
        # Need to override this to get rid of spines.
        return self._xaxis_transform

    def get_yaxis_transform(self, which="grid"):
        # Need to override this to get rid of spines.
        return self._yaxis_transform

    def _gen_axes_spines(self):
        # In axes coordinate system
        spines = OrderedDict(
            [
                (
                    "stereographic",
                    Spine.arc_spine(
                        axes=self,
                        spine_type="top",
                        center=(0.5, 0.5),
                        radius=0.5,
                        theta1=0,
                        theta2=360,
                    ),
                ),
                ("start", Spine.linear_spine(self, "left")),
                ("end", Spine.linear_spine(self, "right")),
            ]
        )
        spines["stereographic"].set_transform(self.transWedge + self.transAxes)
        spines["start"].set_transform(self._yaxis_transform)
        spines["end"].set_transform(self._yaxis_transform)
        return spines

    def _gen_axes_patch(self):
        # In axes coordinate system
        return Wedge(center=(0.5, 0.5), r=0.5, theta1=0, theta2=360)

    def get_data_ratio(self):
        return 1

    def can_pan(self):
        return False

    def can_zoom(self):
        return False

    def _restrict_to_fundamental_sector(self, fs):
        vertices = fs.vertices
        is_vz = np.isclose(Vector3d.zvector().dot(vertices).data, 1)
        azimuth = vertices[~is_vz].azimuth.data
        azimuth_min, azimuth_max = azimuth.min(), azimuth.max()
        polar = vertices.polar.data
        self.viewLim.x0 = azimuth_min
        self.viewLim.x1 = azimuth_max
        self.viewLim.y0 = polar.min()
        self.viewLim.y1 = polar.max()

    def set_azimuth_range(self, amin, amax):
        self.viewLim.x0 = np.deg2rad(amin)
        self.viewLim.x1 = np.deg2rad(amax)

    def set_polar_max(self, pmax):
        self.viewLim.y1 = np.deg2rad(pmax)

    def draw(self, renderer):
        self._unstale_viewLim()

        azimuth_min, azimuth_max = np.rad2deg(self.viewLim.intervalx)
        polar_min, polar_max = self.viewLim.intervaly

        center = self.transWedge.transform((0.5, 0.5))
        self.patch.set_center(center)
        self.patch.set_theta1(azimuth_min)
        self.patch.set_theta2(azimuth_max)

        edge, _ = self.transWedge.transform((1, 0))
        radius = edge - center[0]
        width = min(radius * (polar_max - polar_min) / polar_max, radius)
        self.patch.set_radius(radius)
        self.patch.set_width(width)

        is_full_circle = abs(abs(azimuth_max - azimuth_min) - 360) < 1e-12
        visible = not is_full_circle
        self.spines["start"].set_visible(visible)
        self.spines["end"].set_visible(visible)

        if visible:
            yaxis_text_transform = self._yaxis_transform
        else:
            yaxis_text_transform = self.transData
        if self._yaxis_text_transform != yaxis_text_transform:
            self._yaxis_text_transform.set(yaxis_text_transform)
            self.yaxis.reset_ticks()
            self.yaxis.set_clip_path(self.patch)

        super().draw(renderer)

    def plot(self, *args, **kwargs):
        """Plot vectors as scatter points or draw lines between them.

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
        new_kwargs = dict(clip_on=False, linewidth=2, color="k", linestyle="-")
        out = self._prepare_to_call_inherited_method(args, kwargs, new_kwargs)
        if out is None:
            return
        else:
            azimuth, polar, _, updated_kwargs = out

        super().plot(azimuth, polar, **updated_kwargs)

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
        if out is None:
            return
        else:
            azimuth, polar, visible, updated_kwargs = out

        # Color(s) and size(s)
        c = updated_kwargs.pop("c", "C0")
        c = _get_array_of_values(value=c, visible=visible)
        s = updated_kwargs.pop("s", None)
        if s is not None:
            s = _get_array_of_values(value=s, visible=visible)

        super().scatter(azimuth, polar, c=c, s=s, **updated_kwargs)

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
        if out is None:
            return
        else:
            azimuth, polar, _, updated_kwargs = out
        super().text(azimuth, polar, **updated_kwargs)

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
        self._set_lim_and_transforms()

    @property
    def pole(self):
        """Projection pole, either -1 or 1, where -1 (1) means the
        projection point of the stereographic transform is the south
        (north) pole [00-1] ([001]), i.e. only vectors with z > 0 (z <
        0) are plotted.

        Derived from :attr:`hemisphere`.
        """
        return {"upper": -1, "lower": 1}[self.hemisphere]

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
        Axes.text(self, 0.75 * np.pi, np.pi / 2, s=self.hemisphere, **new_kwargs)

    def azimuth_grid(self, resolution=None):
        """Set the azimuth grid resolution in degrees.

        Parameters
        ----------
        resolution : float, optional
            Azimuth grid resolution in degrees. Default is 15 degrees.
            This can also be set upon initialization of the axes by
            passing `azimuth_resolution` to `subplot_kw`.

        See Also
        --------
        polar_grid
        matplotlib.axes.Axes.grid
        """
        if resolution is not None:
            self._azimuth_resolution = resolution
        grid = np.arange(0, self._azimuth_cap, np.deg2rad(self._azimuth_resolution))
        self.set_xticks(grid)

    def polar_grid(self, resolution=None):
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
        grid = np.arange(0, self._polar_cap, np.deg2rad(self._polar_resolution))
        self.set_yticks(grid)

    def _set_label(self, x, y, label, **kwargs):
        bbox_dict = dict(boxstyle="round, pad=0.1", fc="w", ec="w")
        new_kwargs = dict(ha="center", va="center", bbox=bbox_dict)
        new_kwargs.update(kwargs)
        super().text(x=x, y=y, s=label, **new_kwargs)

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
        # z label position
        if self.pole == -1:
            z_pos = (0, 0)
        else:  # == 1
            z_pos = (0, np.pi)
        pos = [(0, self._polar_cap), (self._polar_cap,) * 2, z_pos]
        for (x, y), label in zip(pos, [xlabel, ylabel, zlabel]):
            if label not in [None, False]:
                self._set_label(x=x, y=y, label=label, **kwargs)

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
        azimuth, polar = self._pretransform_input(args)

        # Exclude vectors not visible in this hemisphere before creating
        # circles
        visible = self._visible_in_hemisphere(polar)
        if np.count_nonzero(visible) == 0:  # No circles to draw
            return
        else:
            azimuth = azimuth[visible]
            polar = polar[visible]

        # Get set of `steps` vectors delineating a circle per vector
        v = Vector3d.from_polar(azimuth=azimuth, polar=polar)
        circles = v.get_circle(opening_angle=opening_angle, steps=steps).unit

        # Enable using one color per circle
        color = kwargs.pop("color", "C0")
        color2 = _get_array_of_values(value=color, visible=visible)

        # Set above which elements circles will appear (zorder)
        new_kwargs = dict(zorder=ZORDER["draw_circle"], clip_on=False)
        for k, v in new_kwargs.items():
            kwargs.setdefault(k, v)

        hemisphere = self.hemisphere
        polar_cap = self._polar_cap
        for i, c in enumerate(circles):
            a, p = _sort_coords_by_shifted_bools(
                hemisphere=hemisphere,
                polar_cap=polar_cap,
                azimuth=c.azimuth.data,
                polar=c.polar.data,
            )
            super().plot(a, p, color=color2[i], **kwargs)

    @staticmethod
    def _pretransform_input(values):
        """Return arrays of azimuth and polar angles from input data.

        Parameters
        ----------
        values : tuple of numpy.ndarray or Vector3d
            Spherical coordinates (azimuth, polar) or vectors. If
            spherical coordinates are given, they are assumed to
            describe unit vectors.

        Returns
        -------
        azimuth : numpy.ndarray
            Azimuth coordinates of unit vectors.
        polar : numpy.ndarray
            Polar coordinates of unit vectors.
        """
        if len(values) == 2:
            azimuth = np.asarray(values[0])
            polar = np.asarray(values[1])
        else:
            try:
                value = values[0].flatten().unit
                azimuth = value.azimuth.data
                polar = value.polar.data
            except (ValueError, AttributeError):
                raise ValueError(
                    "Accepts only one (Vector3d) or two (azimuth, polar) input "
                    "arguments"
                )
        return azimuth, polar

    def _visible_in_hemisphere(self, polar):
        """Return a boolean array describing whether the vector which
        the polar angles belong to are visible in the current
        hemisphere.

        Parameters
        ----------
        polar : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Boolean array with True for the polar angles corresponding
            to vectors visible in this hemisphere.
        """
        return _visible_in_hemisphere(self.hemisphere, self._polar_cap, polar)

    def _prepare_to_call_inherited_method(self, args, kwargs, new_kwargs=None):
        """Prepare arguments and keyword arguments passed to methods in
        :class:`StereographicPlot` inherited from
        :class:`matplotlib.axes.Axes`.

        Parameters
        ----------
        args
            Any arguments passed to the :class:`StereographicPlot` method.
        kwargs : dict
            Any arguments passed to the :class:`StereographicPlot` method.
        new_kwargs : dict
            Any default keyword arguments to be passed to the inherited
            method.

        Returns
        -------
        azimuth : numpy.ndarray
        polar : numpy.ndarray
        visible : numpy.ndarray
        updated_kwargs : dict
        """
        updated_kwargs = kwargs
        for k, v in new_kwargs.items():
            updated_kwargs.setdefault(k, v)

        azimuth, polar = self._pretransform_input(args)
        visible = self._visible_in_hemisphere(polar)
        if np.count_nonzero(visible) == 0:
            return
        else:
            azimuth = azimuth[visible]
            polar = polar[visible]

        return azimuth, polar, visible, updated_kwargs


register_projection(StereographicPlot)


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
        value = [value,] * n
    return np.asarray(value)[visible]


def _visible_in_hemisphere(hemisphere, polar_cap, polar):
    """Return a boolean array describing whether the vector which the
    polar angles belong to are visible in the current hemisphere.

    Parameters
    ----------
    hemisphere : str
    polar_cap : float
    polar : numpy.ndarray

    Returns
    -------
    numpy.ndarray
        Boolean array with True for polar angles corresponding to
        vectors visible in this hemisphere.
    """
    # TODO: Use SphericalRegion, possibly defined by a FundamentalSector
    return polar <= polar_cap if hemisphere == "upper" else polar > polar_cap


def _sort_coords_by_shifted_bools(hemisphere, polar_cap, azimuth, polar):
    """Shift azimuth and polar coordinate arrays so that the ones
    corresponding to vectors visible in this hemisphere are shifted to
    the start of the arrays.

    Used in :meth:`StereographicPlot.draw_circle`.

    Parameters
    ----------
    hemisphere : str
    polar_cap : float
    azimuth : numpy.ndarray
    polar : numpy.ndarray

    Returns
    -------
    azimuth : numpy.ndarray
    polar : numpy.ndarray
    """
    visible = _visible_in_hemisphere(
        hemisphere=hemisphere, polar_cap=polar_cap, polar=polar
    )
    indices = np.asarray(visible != visible[0]).nonzero()[0]
    if indices.size != 0:
        to_shift = indices[-1] + 1
        azimuth = np.roll(azimuth, shift=-to_shift)
        polar = np.roll(polar, shift=-to_shift)
    return azimuth, polar


class _WedgeBbox(mtransforms.Bbox):
    """
    Transform (theta, r) wedge Bbox into axes bounding box.
    Parameters
    ----------
    center : (float, float)
        Center of the wedge
    viewLim : `~matplotlib.transforms.Bbox`
        Bbox determining the boundaries of the wedge
    originLim : `~matplotlib.transforms.Bbox`
        Bbox determining the origin for the wedge, if different from *viewLim*
    """

    def __init__(self, center, viewLim, originLim, **kwargs):
        super().__init__([[0, 0], [1, 1]], **kwargs)
        self._center = center
        self._viewLim = viewLim
        self._originLim = originLim
        self.set_children(viewLim, originLim)

    __str__ = mtransforms._make_str_method("_center", "_viewLim", "_originLim")

    def get_points(self):
        # docstring inherited
        if self._invalid:
            points = self._viewLim.get_points().copy()

            # Scale angular limits to work with Wedge
            points[:, 0] = np.rad2deg(points[:, 0])

            # Scale radial limits to match axes limits
            rscale = 0.5 / points[1, 1]
            points[:, 1] *= rscale
            width = min(points[1, 1] - points[0, 1], 0.5)

            # Generate bounding box for wedge
            wedge = Wedge(
                self._center, points[1, 1], points[0, 0], points[1, 0], width=width
            )
            self.update_from_path(wedge.get_path())

            # Ensure equal aspect ratio
            w, h = self._points[1] - self._points[0]
            deltah = max(w - h, 0) / 2
            deltaw = max(h - w, 0) / 2
            self._points += np.array([[-deltaw, -deltah], [deltaw, deltah]])

            self._invalid = False

        return self._points
