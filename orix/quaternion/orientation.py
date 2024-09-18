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

from __future__ import annotations

from typing import Optional, Tuple, Union
import warnings

import dask.array as da
from dask.diagnostics import ProgressBar
from diffpy.structure import Structure
from matplotlib.gridspec import SubplotSpec
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation

from orix.quaternion.misorientation import Misorientation
from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C1, Symmetry, _get_unique_symmetry_elements
from orix.vector import Miller, Vector3d


class Orientation(Misorientation):
    r"""Orientations represent misorientations away from a reference of
    identity and have only one associated symmetry.

    Orientations :math:`O` support binary subtraction, producing a
    misorientation :math:`M`. That is, to compute the misorientation
    from :math:`O_1` to :math:`O_2`, call :code:`O_2 - O_1`.

    In orix, orientations and misorientations are distinguished from
    rotations only by the inclusion of a notion of symmetry. Consider
    the following example:

    .. image:: /_static/img/orientation.png
       :width: 200px
       :alt: Two objects with two different rotations each. The square,
             with four-fold symmetry, has the same orientation in both
             cases.
       :align: center

    Both objects have undergone the same *rotations* with respect to the
    reference. However, because the square has four-fold symmetry, it is
    indistinguishable in both cases, and hence has the same orientation.
    """

    # -------------------------- Properties -------------------------- #

    @property
    def symmetry(self) -> Symmetry:
        """Symmetry."""
        return self._symmetry[1]

    @symmetry.setter
    def symmetry(self, value: Symmetry):
        if not isinstance(value, Symmetry):
            raise TypeError("Value must be an instance of orix.quaternion.Symmetry.")
        self._symmetry = (C1, value)

    @property
    def unit(self) -> Orientation:
        """Return unit orientations."""
        O = super().unit
        O.symmetry = self.symmetry
        return O

    # ------------------------ Dunder methods ------------------------ #

    def __invert__(self) -> Orientation:
        O = super().__invert__()
        O.symmetry = self.symmetry
        return O

    def __neg__(self) -> Orientation:
        O = super().__neg__()
        O.symmetry = self.symmetry
        return O

    def __repr__(self) -> str:
        """String representation."""
        data = np.array_str(self.data, precision=4, suppress_small=True)
        return f"{self.__class__.__name__} {self.shape} {self.symmetry.name}\n{data}"

    def __sub__(self, other: Orientation) -> Misorientation:
        if isinstance(other, Orientation):
            # Call to Object3d.squeeze() doesn't carry over symmetry
            M = Misorientation(self * ~other).squeeze()
            M.symmetry = (self.symmetry, other.symmetry)
            return M.map_into_symmetry_reduced_zone()
        return NotImplemented

    # ------------------------ Class methods ------------------------- #

    @classmethod
    def from_euler(
        cls,
        euler: Union[np.ndarray, tuple, list],
        symmetry: Optional[Symmetry] = None,
        direction: str = "lab2crystal",
        degrees: bool = False,
    ) -> Orientation:
        """Create orientations from sets of Euler angles
        :cite:`rowenhorst2015consistent`.

        Parameters
        ----------
        euler
            Euler angles in radians (``degrees=False``) or in degrees
            (``degrees=True``) in the Bunge convention.
        symmetry
            Symmetry of orientation(s). If not given (default), no
            symmetry is set.
        direction
            ``"lab2crystal"`` (default) or ``"crystal2lab"``.
            ``"lab2crystal"`` is the Bunge convention. If ``"MTEX"`` is
            provided then the direction is ``"crystal2lab"``.
        degrees
            If ``True``, the given angles are assumed to be in degrees.
            Default is ``False``.

        Returns
        -------
        O
            Orientations.
        """
        O = super().from_euler(euler, direction=direction, degrees=degrees)
        if symmetry:
            O.symmetry = symmetry
        return O

    @classmethod
    def from_align_vectors(
        cls,
        other: Miller,
        initial: Vector3d,
        weights: Optional[np.ndarray] = None,
        return_rmsd: bool = False,
        return_sensitivity: bool = False,
    ) -> Union[
        Orientation,
        Tuple[Orientation, float],
        Tuple[Orientation, np.ndarray],
        Tuple[Orientation, float, np.ndarray],
    ]:
        """Create an estimated orientation to optimally align vectors in
        the crystal and sample reference frames.

        This method wraps
        :meth:`~scipy.spatial.transform.Rotation.align_vectors`. See
        that method for further explanations of parameters and returns.

        Parameters
        ----------
        other
            Crystal directions of shape ``(n,)`` in the crystal
            reference frame.
        initial
            Sample directions of shape ``(n,)`` in the sample reference
            frame.
        weights
            Relative importance of the different vectors.
        return_rmsd
            Whether to return the (weighted) root mean square distance
            between ``other`` and ``initial`` after alignment. Default
            is ``False``.
        return_sensitivity
            Whether to return the sensitivity matrix. Default is
            ``False``.

        Returns
        -------
        estimated_orientation
            Best estimate of the orientation that transforms ``initial``
            to ``other``. The symmetry of the orientation is inferred
            from the point group of the phase of ``other``, if given.
        rmsd
            Returned when ``return_rmsd=True``.
        sensitivity
            Returned when ``return_sensitivity=True``.

        Raises
        ------
        ValueError
            If ``other`` is not a Miller instance.

        Examples
        --------
        >>> from orix.quaternion import Orientation
        >>> from orix.vector import Vector3d, Miller
        >>> from orix.crystal_map import Phase
        >>> t = Miller(uvw=[[0, 1, 0], [1, 0, 0]], phase=Phase(point_group="m-3m"))
        >>> v_sample = Vector3d([[0, -1, 0], [0, 0, 1]])
        >>> O = Orientation.from_align_vectors(t, v_sample)
        >>> O * v_sample
        Vector3d (2,)
        [[0. 1. 0.]
         [1. 0. 0.]]
        """
        if not isinstance(other, Miller):
            raise ValueError(
                f"Argument other must be of type Miller, but has type {type(other)}"
            )

        out = Rotation.from_align_vectors(
            other=other,
            initial=initial,
            weights=weights,
            return_rmsd=return_rmsd,
            return_sensitivity=return_sensitivity,
        )
        out = list(out)
        out[0] = cls(out[0].data)

        try:
            out[0].symmetry = other.phase.point_group
        except (AttributeError, ValueError):
            pass

        return out[0] if len(out) == 1 else tuple(out)

    @classmethod
    def from_matrix(
        cls, matrix: np.ndarray, symmetry: Optional[Symmetry] = None
    ) -> Orientation:
        """Create orientations from orientation matrices
        :cite:`rowenhorst2015consistent`.

        Parameters
        ----------
        matrix
            Array of orientation matrices.
        symmetry
            Symmetry of orientation(s). If not given (default), no
            symmetry is set.

        Returns
        -------
        O
            Orientations.
        """
        O = super().from_matrix(matrix)
        if symmetry:
            O.symmetry = symmetry
        return O

    @classmethod
    def from_axes_angles(
        cls,
        axes: Union[np.ndarray, Vector3d, tuple, list],
        angles: Union[np.ndarray, tuple, list, float],
        symmetry: Optional[Symmetry] = None,
        degrees: bool = False,
    ) -> Orientation:
        """Create orientations from axis-angle pairs
        :cite:`rowenhorst2015consistent`.

        Parameters
        ----------
        axes
            Axes of rotation.
        angles
            Angles of rotation in radians (``degrees=False``) or degrees
            (``degrees=True``).
        symmetry
            Symmetry of orientations. If not given (default), no
            symmetry is set.
        degrees
            If ``True``, the given angles are assumed to be in degrees.
            Default is ``False``.

        Returns
        -------
        O
            Orientation(s).

        Examples
        --------
        >>> from orix.quaternion import Orientation, symmetry
        >>> O = Orientation.from_axes_angles((0, 0, -1), 90, symmetry.Oh, degrees=True)
        >>> O
        Orientation (1,) m-3m
        [[ 0.7071  0.      0.     -0.7071]]
        """
        O = super().from_axes_angles(axes, angles, degrees)
        if symmetry:
            O.symmetry = symmetry
        return O

    @classmethod
    def from_scipy_rotation(
        cls, rotation: SciPyRotation, symmetry: Optional[Symmetry] = None
    ) -> Orientation:
        """Return orientation(s) from
        :class:`scipy.spatial.transform.Rotation`.

        Parameters
        ----------
        rotation
            SciPy rotation(s).
        symmetry
            Crystal symmetry. If not given, the returned orientation(s)
            is given only the identity symmetry operation, *1* (*C1*).

        Returns
        -------
        O
            Orientations.

        Notes
        -----
        The SciPy rotation is inverted to be consistent with the orix
        framework of passive rotations.

        Examples
        --------
        >>> from orix.quaternion import Orientation, symmetry
        >>> from orix.vector import Vector3d
        >>> from scipy.spatial.transform import Rotation as SciPyRotation
        >>> R_scipy = SciPyRotation.from_euler("ZXZ", [90, 0, 0], degrees=True)
        >>> O = Orientation.from_scipy_rotation(R_scipy, symmetry.Oh)
        >>> v = [1, 1, 0]
        >>> R_scipy.apply(v)
        array([-1.,  1.,  0.])
        >>> O * Vector3d(v)
        Vector3d (1,)
        [[ 1. -1.  0.]]
        """
        O = super().from_scipy_rotation(rotation)
        if symmetry:
            O.symmetry = symmetry
        return O

    @classmethod
    def random(
        cls, shape: Union[int, tuple] = 1, symmetry: Optional[Symmetry] = None
    ) -> Orientation:
        """Create random orientations.

        Parameters
        ----------
        shape
            Shape of the orientations.
        symmetry
            Crystal symmetry. If not given, the returned orientation(s)
            is given only the identity symmetry operation, *1* (*C1*).

        Returns
        -------
        O
            Random orientations.
        """
        O = super().random(shape)
        if symmetry:
            O.symmetry = symmetry
        return O

    # --------------------- Other public methods --------------------- #

    def angle_with(self, other: Orientation, degrees: bool = False) -> np.ndarray:
        """Return the smallest symmetry reduced angles of rotation
        transforming the orientations to the other orientations.

        Parameters
        ----------
        other
            Another orientation.
        degrees
            If ``True``, the angles are returned in degrees. Default is
            ``False``.

        Returns
        -------
        angles
            Smallest symmetry reduced angles in radians
            (``degrees=False``) or degrees (``degrees=True``).

        See Also
        --------
        angle_with_outer
        Rotation.angle_with_outer
        """
        dot_products = self.unit.dot(other.unit)
        angles = np.nan_to_num(np.arccos(2 * dot_products**2 - 1))
        if degrees:
            angles = np.rad2deg(angles)
        return angles

    def angle_with_outer(
        self,
        other: Orientation,
        lazy: bool = False,
        chunk_size: int = 20,
        progressbar: bool = True,
        degrees: bool = False,
    ) -> np.ndarray:
        r"""Return the symmetry reduced smallest angle of rotation
        transforming every orientation in this instance to every
        orientation in another instance.

        Parameters
        ----------
        other
            Another orientation.
        lazy
            Whether to perform the computation lazily with Dask. Default
            is ``False``.
        chunk_size
            Number of orientations per axis to include in each iteration
            of the computation. Default is 20. Only applies when
            ``lazy=True``. Increasing this might reduce the computation
            time at the cost of increased memory use.
        progressbar
            Whether to show a progressbar during computation if
            ``lazy=True``. Default is ``True``.
        degrees
            If ``True``, the angles are returned in degrees. Default is
            ``False``.

        Returns
        -------
        angles
            Smallest symmetry reduced angles in radians
            (``degrees=False``) or degrees (``degrees=True``).

        See Also
        --------
        angle_with

        Notes
        -----
        Given two orientations :math:`O_i` and :math:`O_j`, the smallest
        angle is considered as the geodesic distance

        .. math::

            d(O_i, O_j) = \arccos(2(O_i \cdot O_j)^2 - 1),

        where :math:`(O_i \cdot O_j)` is the highest dot product between
        symmetrically equivalent orientations to :math:`O_{i,j}`.

        Examples
        --------
        >>> from orix.quaternion import Orientation, symmetry
        >>> O1 = Orientation.random((5, 3))
        >>> O2 = Orientation.random((6, 2))
        >>> omega1 = O1.angle_with_outer(O2)
        >>> omega1.shape
        (5, 3, 6, 2)
        >>> O1.symmetry = symmetry.Oh
        >>> O2.symmetry = symmetry.Oh
        >>> omega_sym = O1.angle_with_outer(O2)
        >>> np.allclose(omega1.data, omega_sym.data)
        False
        """
        O = self.unit
        if lazy:
            dot_products = O._dot_outer_dask(other, chunk_size=chunk_size)
            # Round because some dot products are slightly above 1
            n_decimals = np.finfo(dot_products.dtype).precision
            dot_products = da.round(dot_products, n_decimals)

            angles_dask = da.arccos(2 * dot_products**2 - 1)
            angles_dask = da.nan_to_num(angles_dask)

            # Create array in memory and overwrite, chunk by chunk
            angles = np.zeros(angles_dask.shape)
            if progressbar:
                with ProgressBar():
                    da.store(sources=angles_dask, targets=angles)
            else:
                da.store(sources=angles_dask, targets=angles)
        else:
            dot_products = O.dot_outer(other)
            angles = np.arccos(2 * dot_products**2 - 1)
            angles = np.nan_to_num(angles)

        if degrees:
            angles = np.rad2deg(angles)

        return angles

    def get_distance_matrix(
        self,
        lazy: bool = False,
        chunk_size: int = 20,
        progressbar: bool = True,
        degrees: bool = False,
    ) -> np.ndarray:
        r"""Return the symmetry reduced smallest angle of rotation
        transforming all these orientations to all the other
        orientations :cite:`johnstone2020density`.

        Parameters
        ----------
        lazy
            Whether to perform the computation lazily with Dask. Default
            is ``False``.
        chunk_size
            Number of orientations per axis to include in each iteration
            of the computation. Default is 20. Only applies when
            ``lazy=True``. Increasing this might reduce the computation
            time at the cost of increased memory use.
        progressbar
            Whether to show a progressbar during computation if
            ``lazy=True``. Default is ``True``.
        degrees
            If ``True``, the angles are returned in degrees. Default is
            ``False``.

        Returns
        -------
        angles
            Symmetry reduced angles in radians (``degrees=False``) or
            degrees (``degrees=True``).

        Notes
        -----
        Given two orientations :math:`g_i` and :math:`g_j`, the smallest
        angle is considered as the geodesic distance

        .. math::

            d(O_i, O_j) = \arccos(2(O_i \cdot O_j)^2 - 1),

        where :math:`(O_i \cdot O_j)` is the highest dot product between
        symmetrically equivalent orientations to :math:`O_{i,j}`.
        """
        angles = self.angle_with_outer(
            self,
            lazy=lazy,
            chunk_size=chunk_size,
            progressbar=progressbar,
            degrees=degrees,
        )
        return angles

    def dot(self, other: Orientation) -> np.ndarray:
        """Return the symmetry reduced dot products of the orientations
        and the other orientations.

        Parameters
        ----------
        other
            Other orientations.

        Returns
        -------
        highest_dot_products
            Symmetry reduced dot products.

        See Also
        --------
        dot_outer

        Examples
        --------
        >>> from orix.quaternion import Orientation, symmetry
        >>> O1 = Orientation.from_axes_angles([0, 0, 1], [0, 45], symmetry.Oh, degrees=True)
        >>> O2 = Orientation.from_axes_angles([0, 0, 1], [45, 90], symmetry.Oh, degrees=True)
        >>> O1.dot(O2)
        array([0.92387953, 0.92387953])
        """
        symmetry = _get_unique_symmetry_elements(self.symmetry, other.symmetry)
        M = other * ~self
        all_dot_products = Rotation(M).dot_outer(symmetry)
        highest_dot_products = np.max(all_dot_products, axis=-1)
        return highest_dot_products

    def dot_outer(self, other: Orientation) -> np.ndarray:
        """Return the symmetry reduced dot products of all orientations
        to all other orientations.

        Parameters
        ----------
        other
            Other orientations.

        Returns
        -------
        highest_dot_products
            Symmetry reduced dot products.

        See Also
        --------
        dot

        Examples
        --------
        >>> from orix.quaternion import Orientation, symmetry
        >>> O1 = Orientation.from_axes_angles([0, 0, 1], [0, 45], symmetry.Oh, degrees=True)
        >>> O2 = Orientation.from_axes_angles([0, 0, 1], [45, 90], symmetry.Oh, degrees=True)
        >>> O1.dot_outer(O2)
        array([[0.92387953, 1.        ],
               [1.        , 0.92387953]])
        """
        symmetry = _get_unique_symmetry_elements(self.symmetry, other.symmetry)
        M = other.outer(~self)
        all_dot_products = Rotation(M).dot_outer(symmetry)
        highest_dot_products = np.max(all_dot_products, axis=-1)
        # need to return axes order so that self is first
        order = tuple(range(self.ndim, self.ndim + other.ndim)) + tuple(
            range(self.ndim)
        )
        return highest_dot_products.transpose(*order)

    def plot_unit_cell(
        self,
        c: str = "tab:blue",
        return_figure: bool = False,
        axes_length: float = 0.5,
        structure: Optional[Structure] = None,
        crystal_axes_loc: str = "origin",
        **arrow_kwargs,
    ) -> plt.Figure:
        """Plot the unit cell orientation, showing the sample and
        crystal reference frames.

        Parameters
        ----------
        c
            Unit cell edge color.
        return_figure
            Return the plotted figure.
        axes_length
            Length of the reference axes in Angstroms, by default 0.5.
        structure
            Structure of the unit cell, only orthorhombic lattices are
            currently supported. If not given, a cubic unit cell with a
            lattice parameter of 2 Angstroms will be plotted.
        crystal_axes_loc
            Plot the crystal reference frame axes at the ``"origin"``
            (default) or ``"center"`` of the plotted cell.
        **arrow_kwargs
            Keyword arguments passed to
            :class:`matplotlib.patches.FancyArrowPatch`, for example
            ``"arrowstyle"``.

        Returns
        -------
        fig
            The plotted figure, returned if ``return_figure=True``.

        Raises
        ------
        ValueError
            If :attr:`size` > 1.
        """
        if self.size > 1:
            raise ValueError("Can only plot a single unit cell, so *size* must be 1")

        from orix.plot.unit_cell_plot import _plot_unit_cell

        fig = _plot_unit_cell(
            self,
            c=c,
            axes_length=axes_length,
            structure=structure,
            crystal_axes_loc=crystal_axes_loc,
            **arrow_kwargs,
        )

        if return_figure:
            return fig

    def in_euler_fundamental_region(self) -> np.ndarray:
        """Euler angles in the fundamental Euler region of the proper
        subgroup.

        The Euler angle ranges of each proper subgroup are given in
        :attr:`~orix.quaternion.Symmetry.euler_fundamental_region`.

        From the procedure in MTEX' :code:`quaternion.project2EulerFR`.

        Returns
        -------
        euler_in_region
            Euler angles in radians.
        """
        pg = self.symmetry.proper_subgroup

        # Symmetrize every orientation by operations of the proper
        # subgroup different from rotation about the c-axis
        O = pg._special_rotation.outer(self)

        alpha, beta, gamma = O.to_euler().T
        gamma = np.mod(gamma, 2 * np.pi / pg._primary_axis_order)

        # Find the first triplet among the symmetrically equivalent ones
        # inside the fundamental region
        max_alpha, max_beta, max_gamma = np.radians(pg.euler_fundamental_region)
        is_inside = (alpha <= max_alpha) * (beta <= max_beta) * (gamma <= max_gamma)
        first_nonzero = np.argmax(is_inside, axis=1)

        euler_in_region = np.column_stack(
            (
                np.choose(first_nonzero, alpha.T),
                np.choose(first_nonzero, beta.T),
                np.choose(first_nonzero, gamma.T),
            )
        )

        return euler_in_region

    def scatter(
        self,
        projection: str = "axangle",
        figure: Optional[plt.Figure] = None,
        position: Union[int, Tuple[int], SubplotSpec] = (1, 1, 1),
        return_figure: bool = False,
        wireframe_kwargs: Optional[dict] = None,
        size: Optional[int] = None,
        direction: Optional[Vector3d] = None,
        figure_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> plt.Figure:
        """Plot orientations in axis-angle space, the Rodrigues
        fundamental zone, or an inverse pole figure (IPF) given a sample
        direction.

        Parameters
        ----------
        projection
            Which orientation space to plot orientations in, either
            "axangle" (default), "rodrigues" or "ipf" (inverse pole
            figure).
        figure
            If given, a new plot axis :class:`~orix.plot.AxAnglePlot` or
            :class:`~orix.plot.RodriguesPlot` is added to the figure in
            the position specified by `position`. If not given, a new
            figure is created.
        position
            Where to add the new plot axis. 121 or (1, 2, 1) places it
            in the first of two positions in a grid of 1 row and 2
            columns. See :meth:`~matplotlib.figure.Figure.add_subplot`
            for further details. Default is (1, 1, 1).
        return_figure
            Whether to return the figure. Default is False.
        wireframe_kwargs
            Keyword arguments passed to
            :meth:`orix.plot.AxAnglePlot.plot_wireframe` or
            :meth:`orix.plot.RodriguesPlot.plot_wireframe`.
        size
            If not given, all orientations are plotted. If given, a
            random sample of this `size` of the orientations is plotted.
        direction
            Sample direction to plot with respect to crystal directions.
            If not given, the out of plane direction, sample Z, is used.
            Only used when plotting IPF(s).
        figure_kwargs
            Dictionary of keyword arguments passed to
            :func:`matplotlib.pyplot.figure` if `figure` is not given.
        **kwargs
            Keyword arguments passed to
            :meth:`orix.plot.AxAnglePlot.scatter`,
            :meth:`orix.plot.RodriguesPlot.scatter`, or
            :meth:`orix.plot.InversePoleFigurePlot.scatter`.

        Returns
        -------
        figure
            Figure with the added plot axis, if ``return_figure=True``.

        See Also
        --------
        orix.plot.AxAnglePlot, orix.plot.RodriguesPlot,
        orix.plot.InversePoleFigurePlot
        """
        if projection.lower() != "ipf":
            figure = super().scatter(
                projection=projection,
                figure=figure,
                position=position,
                return_figure=return_figure,
                wireframe_kwargs=wireframe_kwargs,
                size=size,
                figure_kwargs=figure_kwargs,
                **kwargs,
            )
        else:
            from orix.plot.inverse_pole_figure_plot import (
                _setup_inverse_pole_figure_plot,
            )

            if figure is None:
                # Determine which hemisphere(s) to show
                symmetry = self.symmetry
                sector = symmetry.fundamental_sector
                if np.any(sector.vertices.polar > np.pi / 2):
                    hemisphere = "both"
                else:
                    hemisphere = "upper"

                figure, axes = _setup_inverse_pole_figure_plot(
                    symmetry=symmetry,
                    direction=direction,
                    hemisphere=hemisphere,
                    figure_kwargs=figure_kwargs,
                )
            else:
                axes = np.asarray(figure.axes)

            for ax in axes:
                ax.scatter(self, **kwargs)

        if return_figure:
            return figure

    def inv(self) -> Orientation:
        r"""Return the inverse orientations :math:`O^{-1}`."""
        return self.__invert__()

    # -------------------- Other private methods --------------------- #

    def _dot_outer_dask(self, other: Orientation, chunk_size: int = 20) -> da.Array:
        """Symmetry reduced dot product of every orientation in this
        instance to every orientation in another instance, returned as a
        Dask array.

        Parameters
        ----------
        other
        chunk_size
            Number of orientations per axis to include in each iteration
            of the computation. Default is 20. Increasing this might
            reduce the computation time at the cost of increased memory
            use.

        Returns
        -------
        highest_dot_product

        Notes
        -----
        To read the dot products array `dparr` into memory, do
        `dp = dparr.compute()`.
        """
        symmetry = _get_unique_symmetry_elements(self.symmetry, other.symmetry)
        M = other._outer_dask(~self, chunk_size=chunk_size)

        # Summation subscripts
        str1 = "abcdefghijklmnopqrstuvwxy"[: M.ndim]
        str2 = "z" + str1[-1]  # Last elements have shape (4,)
        sum_over = f"{str1},{str2}->{str1[:-1] + str2[0]}"

        warnings.filterwarnings("ignore", category=da.PerformanceWarning)

        all_dot_products = da.einsum(sum_over, M, symmetry.data)
        highest_dot_product = da.max(abs(all_dot_products), axis=-1)

        return highest_dot_product
