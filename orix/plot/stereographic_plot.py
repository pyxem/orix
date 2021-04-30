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

from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.axis import XAxis, YAxis
from matplotlib.patches import Circle
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D, Affine2DBase, BboxTransformTo, Transform
import numpy as np

from orix.projections import InverseStereographicProjection, StereographicProjection
from orix.plot._symmetry_marker import (
    TwoFoldMarker,
    ThreeFoldMarker,
    FourFoldMarker,
    SixFoldMarker,
)
from orix.vector import Vector3d


class StereographicTransform(Transform):
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


class InvertedStereographicTransform(Transform):
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


class StereographicAffine(Affine2DBase):
    def __init__(self, pole=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pole = pole

    def get_matrix(self):
        # Only recompute if self._invalid is True?
        if self._invalid:
            pole = self.pole
            st = StereographicTransform(pole=pole)
            xscale, _ = st.transform((0, np.pi / 2))
            _, yscale = st.transform((np.pi / 2, np.pi / 2))
            scales = (0.5 / xscale, 0.5 / yscale)
            self._mtx = Affine2D().scale(*scales).translate(0.5, 0.5)
            self._inverted = None
            self._invalid = 0
        return self._mtx


ZORDER = dict(text=6, scatter=5, symmetry_marker=4, draw_circle=3)


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

    # TODO: Extend by taking inspiration from matplotlib.projections.polar:
    #  https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/projections/polar.py

    # See this Matplotlib tutorial for explanations of methods:
    # https://matplotlib.org/stable/gallery/misc/custom_projection.html

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
        # Set ratio of y-unit to x-unit by adjusting the physical
        # dimension of the Axes (box), and centering the anchor (C)
        self.set_aspect("equal", adjustable="box", anchor="C")
        self.clear()

    def _init_axis(self):
        # Need to override these to get rid of spines
        self.xaxis = XAxis(self)
        self.yaxis = YAxis(self)
        self.spines["stereographic"].register_axis(self.yaxis)
        self._update_transScale()

    def clear(self):
        super().clear()

        self.xaxis.set_ticks_position("none")
        self.yaxis.set_ticks_position("none")
        self.xaxis.set_tick_params(label1On=False)
        self.yaxis.set_tick_params(label1On=False)

        self.polar_grid()
        self.azimuth_grid()
        self.grid(rcParams["axes.grid"])

        self.set_xlim(0, self._azimuth_cap)
        self.set_ylim(0, self._polar_cap)

    def _set_lim_and_transforms(self):
        self.transProjection = StereographicTransform(pole=self.pole)
        self.transAffine = StereographicAffine(pole=self.pole)
        self.transAxes = BboxTransformTo(self.bbox)

        self.transData = self.transProjection + self.transAffine + self.transAxes

        self._xaxis_pretransform = Affine2D().scale(1, self._polar_cap)
        self._xaxis_transform = self._xaxis_pretransform + self.transData

        self._yaxis_pretransform = Affine2D().scale(self._azimuth_cap, 1)
        self._yaxis_transform = self._yaxis_pretransform + self.transData

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
        return {"stereographic": Spine.circular_spine(self, (0.5, 0.5), 0.5)}

    @staticmethod
    def _gen_axes_patch():
        return Circle((0.5, 0.5), 0.5)

    @staticmethod
    def get_data_ratio():
        return 1

    @staticmethod
    def can_pan():
        return False

    @staticmethod
    def can_zoom():
        # TODO: Implement zoom (https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/projections/polar.py#L1437)
        return False

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
        for k, v in new_kwargs.items():
            kwargs.setdefault(k, v)

        azimuth, polar = self._pretransform_input(args)
        visible = self._visible_in_hemisphere(polar)
        if np.count_nonzero(visible) == 0:
            return
        else:
            azimuth = azimuth[visible]
            polar = polar[visible]

        # Color(s) and size(s)
        c = kwargs.pop("c", "C0")
        c = _get_array_of_values(value=c, visible=visible)
        s = kwargs.pop("s", None)
        if s is not None:
            s = _get_array_of_values(value=s, visible=visible)

        super().scatter(azimuth, polar, c=c, s=s, **kwargs)

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
        for k, v in new_kwargs.items():
            kwargs.setdefault(k, v)

        azimuth, polar = self._pretransform_input(args)
        visible = self._visible_in_hemisphere(polar)
        if np.count_nonzero(visible) == 0:
            return
        else:
            azimuth = azimuth[visible]
            polar = polar[visible]

        super().text(azimuth, polar, **kwargs)

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
            Aziumuth grid resolution in degrees. Default is 15 degrees.
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
            Azimuth coordiantes of unit vectors.
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
