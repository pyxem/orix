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

import matplotlib.pyplot as plt
import numpy as np

from orix.base import check, Object3d
from orix.scalar import Scalar


def check_vector(obj):
    return check(obj, Vector3d)


class Vector3d(Object3d):
    """Vector base class.

    Vectors support the following mathematical operations:

    - Unary negation.
    - Addition to other vectors, scalars, numbers, and compatible
      array-like objects.
    - Subtraction to and from the above.
    - Multiplication to scalars, numbers, and compatible array-like objects.
    - Division by the same as multiplication. Division by a vector is not
      defined in general.

    Examples
    --------
    >>> import numpy as np
    >>> from orix.vector import Vector3d
    >>> v = Vector3d((1, 2, 3))
    >>> w = Vector3d(np.array([[1, 0, 0], [0, 1, 1]]))
    >>> w.x
    Scalar (2,)
    [1 0]
    >>> v.unit
    Vector3d (1,)
    [[ 0.2673  0.5345  0.8018]]
    >>> -v
    Vector3d (1,)
    [[-1 -2 -3]]
    >>> v + w
    Vector3d (2,)
    [[2 2 3]
     [1 3 4]]
    >>> w - (2, -3)
    Vector3d (2,)
    [[-1 -2 -2]
     [ 3  4  4]]
    >>> 3 * v
    Vector3d (1,)
    [[3 6 9]]
    >>> v / 2
    Vector3d (1,)
    [[0.5 1.0 1.5]]
    >>> v / (2, -2)
    Vector3d (1,)
    [[0.5 1.0 1.5]
     [-0.5 -1.0 -1.5]]
    """

    dim = 3

    @property
    def x(self):
        """Scalar : This vector's x data."""
        return Scalar(self.data[..., 0])

    @x.setter
    def x(self, value):
        self.data[..., 0] = value

    @property
    def y(self):
        """Scalar : This vector's y data."""
        return Scalar(self.data[..., 1])

    @y.setter
    def y(self, value):
        self.data[..., 1] = value

    @property
    def z(self):
        """Scalar : This vector's z data."""
        return Scalar(self.data[..., 2])

    @z.setter
    def z(self, value):
        self.data[..., 2] = value

    @property
    def xyz(self):
        """tuple of ndarray : This vector's components, useful for plotting."""
        return self.x.data, self.y.data, self.z.data

    @property
    def _tuples(self):
        """set of tuple : the set of comparable vectors."""
        s = self.flatten()
        tuples = set([tuple(d) for d in s.data])
        return tuples

    @property
    def perpendicular(self):
        if np.any(self.x.data == 0) and np.any(self.y.data == 0):
            if np.any(self.z.data == 0):
                raise ValueError("No vectors are perpendicular")
            return Vector3d.xvector()
        x = -self.y.data
        y = self.x.data
        z = np.zeros_like(x)
        return Vector3d(np.stack((x, y, z), axis=-1))

    @property
    def radial(self):
        """Radial spherical coordinate, i.e. the distance from a point
        on the sphere to the origin, according to the ISO 31-11 standard
        [SphericalWolfram]_.

        Returns
        -------
        Scalar
        """
        return Scalar(
            np.sqrt(
                self.data[..., 0] ** 2 + self.data[..., 1] ** 2 + self.data[..., 2] ** 2
            )
        )

    @property
    def azimuth(self):
        r"""Azimuth spherical coordinate, i.e. the angle
        :math:`\phi \in [0, 2\pi]` from the positive z-axis to a point
        on the sphere, according to the ISO 31-11 standard
        [SphericalWolfram]_.

        Returns
        -------
        Scalar
        """
        azimuth = np.arctan2(self.data[..., 1], self.data[..., 0])
        azimuth += (azimuth < 0) * 2 * np.pi
        return Scalar(azimuth)

    @property
    def polar(self):
        r"""Polar spherical coordinate, i.e. the angle
        :math:`\theta \in [0, \pi]` from the positive z-axis to a point
        on the sphere, according to the ISO 31-11 standard
        [SphericalWolfram]_.

        Returns
        -------
        Scalar
        """
        return Scalar(np.arccos(self.data[..., 2] / self.radial.data))

    def __neg__(self):
        return self.__class__(-self.data)

    def __add__(self, other):
        if isinstance(other, Vector3d):
            return self.__class__(self.data + other.data)
        elif isinstance(other, Scalar):
            return self.__class__(self.data + other.data[..., np.newaxis])
        elif isinstance(other, (int, float)):
            return self.__class__(self.data + other)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(self.data + other[..., np.newaxis])
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Scalar):
            return self.__class__(other.data[..., np.newaxis] + self.data)
        elif isinstance(other, (int, float)):
            return self.__class__(other + self.data)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(other[..., np.newaxis] + self.data)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector3d):
            return self.__class__(self.data - other.data)
        elif isinstance(other, Scalar):
            return self.__class__(self.data - other.data[..., np.newaxis])
        elif isinstance(other, (int, float)):
            return self.__class__(self.data - other)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(self.data - other[..., np.newaxis])
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Scalar):
            return self.__class__(other.data[..., np.newaxis] - self.data)
        elif isinstance(other, (int, float)):
            return self.__class__(other - self.data)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(other[..., np.newaxis] - self.data)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Vector3d):
            raise ValueError(
                "Multiplying one vector with another is ambiguous. "
                "Try `.dot` or `.cross` instead."
            )
        elif isinstance(other, Scalar):
            return self.__class__(self.data * other.data[..., np.newaxis])
        elif isinstance(other, (int, float)):
            return self.__class__(self.data * other)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(self.data * other[..., np.newaxis])
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Scalar):
            return self.__class__(other.data[..., np.newaxis] * self.data)
        elif isinstance(other, (int, float)):
            return self.__class__(other * self.data)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(other[..., np.newaxis] * self.data)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Vector3d):
            raise ValueError("Dividing vectors is undefined")
        elif isinstance(other, Scalar):
            return self.__class__(self.data / other.data[..., np.newaxis])
        elif isinstance(other, (int, float)):
            return self.__class__(self.data / other)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(self.data / other[..., np.newaxis])
        return NotImplemented

    def __rtruediv__(self, other):
        raise ValueError("Division by a vector is undefined")

    def dot(self, other):
        """The dot product of a vector with another vector.

        Vectors must have compatible shape.

        Returns
        -------
        Scalar

        Examples
        --------
        >>> v = Vector3d((0, 0, 1.0))
        >>> w = Vector3d(((0, 0, 0.5), (0.4, 0.6, 0)))
        >>> v.dot(w)
        Scalar (2,)
        [ 0.5  0. ]
        >>> w.dot(v)
        Scalar (2,)
        [ 0.5  0. ]
        """
        if not isinstance(other, Vector3d):
            raise ValueError("{} is not a vector!".format(other))
        return Scalar(np.sum(self.data * other.data, axis=-1))

    def dot_outer(self, other):
        """The outer dot product of a vector with another vector.

        The dot product for every combination of vectors in `self` and `other`
        is computed.

        Returns
        -------
        Scalar

        Examples
        --------
        >>> v = Vector3d(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))  # shape = (2, )
        >>> w = Vector3d(((0.0, 0.0, 0.5), (0.4, 0.6, 0.0), (0.5, 0.5, 0.5)))  # shape = (3, )
        >>> v.dot_outer(w)
        Scalar (2, 3)
        [[ 0.5  0.   0.5]
         [ 0.   0.4  0.5]]
        >>> w.dot_outer(v)  # shape = (3, 2)
        Scalar (3, 2)
        [[ 0.5  0. ]
         [ 0.   0.4]
         [ 0.5  0.5]]
        """
        dots = np.tensordot(self.data, other.data, axes=(-1, -1))
        return Scalar(dots)

    def cross(self, other):
        """The cross product of a vector with another vector.

        Vectors must have compatible shape for broadcasting to work.

        Returns
        -------
        Vector3d
            The class of 'other' is preserved.

        Examples
        --------
        >>> v = Vector3d(((1, 0, 0), (-1, 0, 0)))
        >>> w = Vector3d((0, 1, 0))
        >>> v.cross(w)
        Vector3d (2,)
        [[ 0  0  1]
         [ 0  0 -1]]
        """
        return other.__class__(np.cross(self.data, other.data))

    @classmethod
    def from_polar(cls, azimuth, polar, radial=1):
        """Create a :class:`~orix.vector.Vector3d` from spherical
        coordinates according to the ISO 31-11 standard
        [SphericalWolfram]_.

        Parameters
        ----------
        azimuth : array_like
            The azimuth angle, in radians.
        polar : array_like
            The polar angle, in radians.
        radial : array_like
            The radial distance. Defaults to 1 to produce unit vectors.

        Returns
        -------
        Vector3d

        References
        ----------
        .. [SphericalWolfram] Weisstein, Eric W. "Spherical Coordinates,"
            *From MathWorld--A Wolfram Web Resource*,
            url: https://mathworld.wolfram.com/SphericalCoordinates.html
        """
        azimuth = np.atleast_1d(azimuth)
        polar = np.atleast_1d(polar)
        sin_polar = np.sin(polar)
        x = np.cos(azimuth) * sin_polar
        y = np.sin(azimuth) * sin_polar
        z = np.cos(polar)
        return radial * cls(np.stack((x, y, z), axis=-1))

    @classmethod
    def zero(cls, shape=(1,)):
        """Returns zero vectors in the specified shape.

        Parameters
        ----------
        shape : tuple

        Returns
        -------
        Vector3d
        """
        return cls(np.zeros(shape + (cls.dim,)))

    @classmethod
    def xvector(cls):
        """Vector3d : a single unit vector parallel to the x-direction."""
        return cls((1, 0, 0))

    @classmethod
    def yvector(cls):
        """Vector3d : a single unit vector parallel to the y-direction."""
        return cls((0, 1, 0))

    @classmethod
    def zvector(cls):
        """Vector3d : a single unit vector parallel to the z-direction."""
        return cls((0, 0, 1))

    def angle_with(self, other):
        """Calculate the angles between vectors in 'self' and 'other'

        Vectors must have compatible shapes for broadcasting to work.

        Returns
        -------
        Scalar
            The angle between the vectors, in radians.
        """
        cosines = np.round(self.dot(other).data / self.norm.data / other.norm.data, 9)
        return Scalar(np.arccos(cosines))

    def rotate(self, axis=None, angle=0):
        """Convenience function for rotating this vector.

        Shapes of 'axis' and 'angle' must be compatible with shape of this
        vector for broadcasting.

        Parameters
        ----------
        axis : Vector3d or array_like, optional
            The axis of rotation. Defaults to the z-vector.
        angle : array_like, optional
            The angle of rotation, in radians.

        Returns
        -------
        Vector3d
            A new vector with entries rotated.

        Examples
        --------
        >>> from math import pi
        >>> v = Vector3d((0, 1, 0))
        >>> axis = Vector3d((0, 0, 1))
        >>> angles = [0, pi/4, pi/2, 3*pi/4, pi]
        >>> v.rotate(axis=axis, angle=angles)
        """
        # Import here to avoid circular import
        from orix.quaternion import Rotation
        from orix.vector.neo_euler import AxAngle

        axis = Vector3d.zvector() if axis is None else axis
        angle = 0 if angle is None else angle
        q = Rotation.from_neo_euler(AxAngle.from_axes_angles(axis, angle))
        return q * self

    def get_nearest(self, x, inclusive=False, tiebreak=None):
        """The vector among x with the smallest angle to this one.

        Parameters
        ----------
        x : Vector3d
        inclusive : bool
            if False (default) vectors exactly parallel to this will not
            be considered.
        tiebreak : Vector3d
            If multiple vectors are equally close to this one,
            `tiebreak` will be used as a secondary comparison. By
            default equal to (0, 0, 1).

        Returns
        -------
        Vector3d
        """
        assert self.size == 1, "`get_nearest` only works for single vectors."
        tiebreak = Vector3d.zvector() if tiebreak is None else tiebreak
        eps = 1e-9 if inclusive else 0.0
        cosines = x.dot(self).data
        mask = np.logical_and(-1 - eps < cosines, cosines < 1 + eps)
        x = x[mask]
        if x.size == 0:
            return Vector3d.empty()
        cosines = cosines[mask]
        verticality = x.dot(tiebreak).data
        order = np.lexsort((cosines, verticality))
        return x[order[-1]]

    def mean(self):
        axis = tuple(range(self.data_dim))
        return self.__class__(self.data.mean(axis=axis))

    def to_polar(self):
        r"""Return the azimuth :math:`\phi`, polar :math:`\theta`, and
        radial :math:`r` spherical coordinates, the angles in radians.
        The coordinates are defined as in the ISO 31-11 standard
        [SphericalWolfram]_.

        Returns
        -------
        azimuth, polar, radial : Scalar
        """
        return self.azimuth, self.polar, self.radial

    def get_circle(self, opening_angle=np.pi / 2, steps=100):
        r"""Get vectors delineating great or small circle(s) with a
        given `opening_angle` about each vector.

        Used for plotting plane traces in stereographic projections.

        Parameters
        ----------
        opening_angle : float or numpy.ndarray, optional
            Opening angle(s) around the vector(s). Default is
            :math:`\pi/2`, giving a great circle. If an array is passed,
            its size must be equal to the number of vectors.
        steps : int, optional
            Number of vectors to describe each circle, default is 100.

        Returns
        -------
        circles : Vector3d
            Vectors delinating circles with the `opening_angle` about
            the vectors.

        Notes
        -----
        A set of `steps` number of vectors equal to each vector is
        rotated twice to obtain a circle: (1) About a perpendicular
        vector to the current vector at `opening_angle` and (2) about
        the current vector in a full circle.
        """
        circles = self.zero((self.size, steps))
        full_circle = np.linspace(0, 2 * np.pi, num=steps, endpoint=True)
        opening_angles = np.ones(self.size) * opening_angle
        for i, (v, oa) in enumerate(zip(self.flatten(), opening_angles)):
            circles[i] = v.rotate(v.perpendicular, oa).rotate(v, full_circle)
        return circles

    def scatter(
        self,
        projection="stereographic",
        figure=None,
        axes_labels=[None, None, None],
        vector_labels=None,
        hemisphere=None,
        show_hemisphere_label=None,
        grid=None,
        grid_resolution=None,
        figure_kwargs=dict(),
        text_kwargs=dict(),
        return_figure=False,
        **kwargs,
    ):
        """Plot vectors in the stereographic projection.

        Parameters
        ----------
        %s
        %s
        %s
        vector_labels : list of str, optional
            Vector text labels, which by default aren't added.
        %s
        %s
        %s
        %s
        %s
        text_kwargs : dict, optional
            Dictionary of keyword arguments passed to
            :func:`~orix.plot.StereographicPlot.text`, which passes
            these on to :meth:`matplotlib.axes.Axes.text`.
        %s
        kwargs : dict, optional
            Keyword arguments passed to
            :func:`~orix.plot.StereographicPlot.scatter`, which passes
            these on to :meth:`matplotlib.axes.Axes.scatter`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure, returned if `return_figure` is True.

        Notes
        -----
        %s

        See Also
        --------
        orix.plot.StereographicPlot
        """
        (
            fig,
            axes,
            hemisphere,
            show_hemisphere_label,
            grid,
            grid_resolution,
        ) = self._setup_plot(
            projection=projection,
            figure=figure,
            hemisphere=hemisphere,
            show_hemisphere_label=show_hemisphere_label,
            grid=grid,
            grid_resolution=grid_resolution,
            figure_kwargs=figure_kwargs,
        )

        # Use methods of the StereographicPlot class
        for i, ax in enumerate(axes):  # Assumes a maximum of two axes
            ax.hemisphere = hemisphere[i]
            ax.scatter(self, **kwargs)
            ax.grid(grid[i])
            ax._stereographic_grid = grid[i]
            ax.azimuth_grid(grid_resolution[0])
            ax.polar_grid(grid_resolution[1])
            ax.set_labels(*axes_labels)
            if show_hemisphere_label:
                ax.show_hemisphere_label()
            if vector_labels is not None:
                for vi, labeli in zip(self, vector_labels):
                    ax.text(vi, s=labeli, **text_kwargs)

        if return_figure:
            return fig

    def draw_circle(
        self,
        projection="stereographic",
        figure=None,
        opening_angle=np.pi / 2,
        steps=100,
        axes_labels=[None, None, None],
        hemisphere=None,
        show_hemisphere_label=None,
        grid=None,
        grid_resolution=None,
        figure_kwargs=dict(),
        return_figure=False,
        **kwargs,
    ):
        r"""Draw great or small circles with a given `opening_angle` to
        to the vectors in the stereographic projection.

        A vector must be present in the current hemisphere for its
        circle to be drawn.

        Parameters
        ----------
        %s
        %s
        opening_angle : float or numpy.ndarray, optional
            Opening angle(s) around the vector(s). Default is
            :math:`\pi/2`, giving a great circle. If an array is passed,
            its size must be equal to the number of vectors.
        steps : int, optional
            Number of vectors to describe each circle, default is 100.
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.plot` to alter the circles'
            appearance.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure, returned if `return_figure` is True.

        Notes
        -----
        %s

        See Also
        --------
        orix.plot.StereographicPlot
        orix.vector.Vector3d.get_circle
        """
        (
            fig,
            axes,
            hemisphere,
            show_hemisphere_label,
            grid,
            grid_resolution,
        ) = self._setup_plot(
            projection=projection,
            figure=figure,
            hemisphere=hemisphere,
            show_hemisphere_label=show_hemisphere_label,
            grid=grid,
            grid_resolution=grid_resolution,
            figure_kwargs=figure_kwargs,
        )

        # Use methods of the StereographicPlot class
        for i, ax in enumerate(axes):  # Assumes a maximum of two axes
            ax.hemisphere = hemisphere[i]
            ax.draw_circle(self, opening_angle=opening_angle, steps=steps, **kwargs)
            ax.grid(grid[i])
            ax._stereographic_grid = grid[i]
            ax.azimuth_grid(grid_resolution[0])
            ax.polar_grid(grid_resolution[1])
            ax.set_labels(*axes_labels)
            if show_hemisphere_label:
                ax.show_hemisphere_label()

        if return_figure:
            return fig

    @staticmethod
    def _setup_plot(
        projection="stereographic",
        figure=None,
        hemisphere=None,
        show_hemisphere_label=None,
        grid=None,
        grid_resolution=None,
        figure_kwargs=dict(),
    ):
        """Set up a stereographic projection plot.

        Parameters
        ----------
        %s
        %s
        %s
        %s
        %s
        %s
        %s

        Returns
        -------
        figure : matplotlib.figure.Figure
        axes : matplotlib.axes.Axes
        hemisphere : tuple of str
        show_hemisphere_label : bool
        grid : list of bool
        grid_resolution : tuple
        """
        if projection.lower() != "stereographic":
            raise NotImplementedError("Stereographic is the only supported projection")

        import orix.plot.stereographic_plot

        if figure is not None:
            axes = figure.axes
            hemisphere = "both" if len(axes) == 2 else axes[0].hemisphere

        # Which hemisphere(s) to plot
        ncols = 1
        hemispheres = ("upper", "lower")
        if hemisphere is None:
            hemisphere = "upper"
        if hemisphere.lower() in hemispheres:
            hemisphere = (hemisphere,)
        elif hemisphere == "both":
            ncols = 2
            hemisphere = hemispheres
            if show_hemisphere_label != False:  # True or None
                show_hemisphere_label = True

        # Create new figure and axis/axes
        if figure is None:
            figure, axes = plt.subplots(
                ncols=ncols, subplot_kw=dict(projection=projection), **figure_kwargs,
            )

        # Make axes iterable
        axes = [axes] if not hasattr(axes, "__iter__") else axes

        if show_hemisphere_label is None:
            show_hemisphere_label = False

        # Whether to plot a grid, and with which resolution
        if grid is None:
            grid = [a._stereographic_grid for a in axes]
            if all(g is None for g in grid):
                grid = [plt.rcParams["axes.grid"],] * ncols
        else:
            grid = [grid,] * ncols
        if grid_resolution is None:
            grid_resolution = (None,) * 2

        return figure, axes, hemisphere, show_hemisphere_label, grid, grid_resolution


PLOT_PROJECTION_DOCSTRING = """projection : str, optional
            Which projection to use. The default is "stereographic", the
            only current option.
    """
PLOT_FIGURE_DOCSTRING = """figure : matplotlib.figure.Figure, optional
            Which figure to plot onto. Default is None, which creates a
            new figure.
    """
PLOT_AXES_LABELS_DOCSTRING = """axes_labels : list of str, optional
            Reference frame axes labels, defaults to [None, None, None].
    """
PLOT_HEMISPHERE_DOCTRING = """hemisphere : str, optional
            Which hemisphere(s) to plot the vectors in, defaults to
            "None", which means "upper" if a new figure is created,
            otherwise adds to the current figure's hemispheres. Options
            are "upper", "lower", and "both", which plots two
            projections side by side.
    """
PLOT_SHOW_HEMISPHERE_LABEL_DOCSTRING = """show_hemisphere_label : bool, optional
            Whether to show hemisphere labels "upper" or "lower".
            Default is True if `hemisphere` is "both", otherwise False.
    """
PLOT_GRID_DOCSTRING = """grid : bool, optional
            Whether to show the azimuth and polar grid. Default is
            whatever `axes.grid` is set to in
            :obj:`matplotlib.rcParams`.
    """
PLOT_GRID_RESOLUTION_DOCSTRING = """grid_resolution : tuple, optional
            Azimuth and polar grid resolution in degrees, as a tuple.
            Default is whatever is default in
            :class:`~orix.plot.StereographicPlot.azimuth_grid` and
            :class:`~orix.plot.StereographicPlot.polar_grid`.
    """
PLOT_RETURN_FIGURE_DOCSTRING = """return_figure : bool, optional
            Whether to return the figure (default is False).
    """
PLOT_FIGURE_KWARGS_DOCSTRING = """figure_kwargs : dict, optional
            Dictionary of keyword arguments passed to
            :func:`matplotlib.pyplot.subplots`.
    """
PLOT_NOT_SO_CUSTOMIZABLE_NOTE_DOCSTRING = """This is a somewhat customizable convenience method which creates
        a figure with axes using :class:`~orix.plot.StereographicPlot`,
        however, it is meant for quick plotting and prototyping. This
        figure and the axes can also be created using Matplotlib
        directly, which is more customizable.
    """
Vector3d.scatter.__doc__ %= (
    PLOT_PROJECTION_DOCSTRING,
    PLOT_FIGURE_DOCSTRING,
    PLOT_AXES_LABELS_DOCSTRING,
    PLOT_HEMISPHERE_DOCTRING,
    PLOT_SHOW_HEMISPHERE_LABEL_DOCSTRING,
    PLOT_GRID_DOCSTRING,
    PLOT_GRID_RESOLUTION_DOCSTRING,
    PLOT_FIGURE_KWARGS_DOCSTRING,
    PLOT_RETURN_FIGURE_DOCSTRING,
    PLOT_NOT_SO_CUSTOMIZABLE_NOTE_DOCSTRING,
)
Vector3d.draw_circle.__doc__ %= (
    PLOT_PROJECTION_DOCSTRING,
    PLOT_FIGURE_DOCSTRING,
    PLOT_AXES_LABELS_DOCSTRING,
    PLOT_HEMISPHERE_DOCTRING,
    PLOT_SHOW_HEMISPHERE_LABEL_DOCSTRING,
    PLOT_GRID_DOCSTRING,
    PLOT_GRID_RESOLUTION_DOCSTRING,
    PLOT_FIGURE_KWARGS_DOCSTRING,
    PLOT_RETURN_FIGURE_DOCSTRING,
    PLOT_NOT_SO_CUSTOMIZABLE_NOTE_DOCSTRING,
)
Vector3d._setup_plot.__doc__ %= (
    PLOT_PROJECTION_DOCSTRING,
    PLOT_FIGURE_DOCSTRING,
    PLOT_HEMISPHERE_DOCTRING,
    PLOT_SHOW_HEMISPHERE_LABEL_DOCSTRING,
    PLOT_GRID_DOCSTRING,
    PLOT_GRID_RESOLUTION_DOCSTRING,
    PLOT_FIGURE_KWARGS_DOCSTRING,
)
