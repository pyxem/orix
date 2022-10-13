# -*- coding: utf-8 -*-
# Copyright 2018-2022 the orix developers
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

from itertools import product as iproduct
from typing import List, Optional, Tuple, Union
import warnings

import dask.array as da
from dask.diagnostics import ProgressBar
from diffpy.structure import Structure
from matplotlib.gridspec import SubplotSpec
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from orix.quaternion.orientation_region import OrientationRegion
from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C1, Symmetry, _get_unique_symmetry_elements
from orix.vector import AxAngle, Miller, NeoEuler, Vector3d


class Misorientation(Rotation):
    r"""Misorientation object.

    Misorientations represent transformations from one orientation,
    :math:`g_1` to another, :math:`g_2`: :math:`g_2 \cdot g_1^{-1}`.

    They have symmetries associated with each of the starting
    orientations.

    Parameters
    ----------
    data
        Quaternions.
    symmetry
        Crystal symmetries.
    """

    _symmetry = (C1, C1)

    def __init__(self, data: np.ndarray, symmetry: Optional[Symmetry] = None):
        super().__init__(data)
        if symmetry:
            self.symmetry = symmetry

    @property
    def symmetry(self) -> Tuple[Symmetry, Symmetry]:
        """Return or set the crystal symmetries.

        Parameters
        ----------
        value : list of Symmetry or 2-tuple of Symmetry
            Crystal symmetries.
        """
        return self._symmetry

    @symmetry.setter
    def symmetry(self, value: Union[List[Symmetry], Tuple[Symmetry, Symmetry]]):
        if not isinstance(value, (list, tuple)):
            raise TypeError("Value must be a 2-tuple of Symmetry objects.")
        if len(value) != 2 or not all(isinstance(s, Symmetry) for s in value):
            raise ValueError("Value must be a 2-tuple of Symmetry objects.")
        self._symmetry = tuple(value)

    def __getitem__(self, key) -> Misorientation:
        mori = super().__getitem__(key)
        mori._symmetry = self._symmetry
        return mori

    def __eq__(self, other):
        v1 = super().__eq__(other)
        if not v1:
            return v1
        else:
            # check symmetries are also equivalent
            v2 = []
            for sym_s, sym_o in zip(self._symmetry, other._symmetry):
                v2.append(sym_s == sym_o)
            return all(v2)

    def reshape(self, *shape) -> Misorientation:
        mori = super().reshape(*shape)
        mori._symmetry = self._symmetry
        return mori

    def flatten(self) -> Misorientation:
        mori = super().flatten()
        mori._symmetry = self._symmetry
        return mori

    def squeeze(self) -> Misorientation:
        mori = super().squeeze()
        mori._symmetry = self._symmetry
        return mori

    def transpose(self, *axes) -> Misorientation:
        mori = super().transpose(*axes)
        mori._symmetry = self._symmetry
        return mori

    def equivalent(self, grain_exchange: bool = False) -> Misorientation:
        r"""Return the equivalent misorientations.

        grain_exchange
            If ``True`` the rotation :math:`g` and :math:`g^{-1}` are
            considered to be identical. Default is ``False``.

        Returns
        -------
        mori
            The equivalent misorientations.
        """
        Gl, Gr = self._symmetry

        if grain_exchange and (Gl._tuples == Gr._tuples):
            orientations = Orientation.stack([self, ~self]).flatten()
        else:
            orientations = Orientation(self)

        equivalent = Gr.outer(orientations.outer(Gl))
        return self.__class__(equivalent).flatten()

    def map_into_symmetry_reduced_zone(self, verbose: bool = False) -> Misorientation:
        """Return equivalent transformations which have the smallest
        angle of rotation as a new misorientation.

        Parameters
        ----------
        verbose
            Whether to print a progressbar. Default is ``False``.

        Returns
        -------
        mori
            A new misorientation object with the assigned symmetry.

        Examples
        --------
        >>> from orix.quaternion.symmetry import C4, C2
        >>> data = np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 0, 0]])
        >>> m = Misorientation(data)
        >>> m.symmetry = (C4, C2)
        >>> m.map_into_symmetry_reduced_zone()
        Misorientation (2,) 4, 2
        [[-0.7071  0.7071  0.      0.    ]
        [ 0.      1.      0.      0.    ]]
        """
        Gl, Gr = self._symmetry
        symmetry_pairs = iproduct(Gl, Gr)
        if verbose:
            symmetry_pairs = tqdm(symmetry_pairs, total=Gl.size * Gr.size)

        orientation_region = OrientationRegion.from_symmetry(Gl, Gr)
        o_inside = self.__class__.identity(self.shape)
        outside = np.ones(self.shape, dtype=bool)
        for gl, gr in symmetry_pairs:
            o_transformed = gl * self[outside] * gr
            o_inside[outside] = o_transformed
            outside = ~(o_inside < orientation_region)
            if not np.any(outside):
                break
        o_inside._symmetry = (Gl, Gr)
        return o_inside

    def __repr__(self):
        """String representation."""
        cls = self.__class__.__name__
        shape = str(self.shape)
        s1, s2 = self._symmetry[0].name, self._symmetry[1].name
        s2 = "" if s2 == "1" else s2
        symm = s1 + (s2 and ", ") + s2
        data = np.array_str(self.data, precision=4, suppress_small=True)
        rep = "{} {} {}\n{}".format(cls, shape, symm, data)
        return rep

    def scatter(
        self,
        projection: str = "axangle",
        figure: Optional[plt.Figure] = None,
        position: Union[int, Tuple[int, int], SubplotSpec] = None,
        return_figure: bool = False,
        wireframe_kwargs: Optional[dict] = None,
        size: Optional[int] = None,
        figure_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> plt.Figure:
        """Plot misorientations in axis-angle space or the Rodrigues
        fundamental zone.

        Parameters
        ----------
        projection
            Which misorientation space to plot misorientations in,
            either ``"axangle"`` (default) or ``"rodrigues"``.
        figure
            If given, a new plot axis :class:`~orix.plot.AxAnglePlot` or
            :class:`~orix.plot.RodriguesPlot` is added to the figure in
            the position specified by ``position``. If not given, a new
            figure is created.
        position
            Where to add the new plot axis. 121 or (1, 2, 1) places it
            in the first of two positions in a grid of 1 row and 2
            columns. See :meth:`~matplotlib.figure.Figure.add_subplot`
            for further details. Default is (1, 1, 1).
        return_figure
            Whether to return the figure. Default is ``False``.
        wireframe_kwargs
            Keyword arguments passed to
            :meth:`orix.plot.AxAnglePlot.plot_wireframe` or
            :meth:`orix.plot.RodriguesPlot.plot_wireframe`.
        size
            If not given, all misorientations are plotted. If given, a
            random sample of this ``size`` of the misorientations is
            plotted.
        figure_kwargs
            Dictionary of keyword arguments passed to
            :func:`matplotlib.pyplot.figure` if ``figure`` is not given.
        **kwargs
            Keyword arguments passed to
            :meth:`orix.plot.AxAnglePlot.scatter` or
            :meth:`orix.plot.RodriguesPlot.scatter`.

        Returns
        -------
        figure
            Figure with the added plot axis, if ``return_figure=True``.

        See Also
        --------
        orix.plot.AxAnglePlot
        orix.plot.RodriguesPlot
        """
        from orix.plot.rotation_plot import _setup_rotation_plot

        figure, ax = _setup_rotation_plot(
            figure=figure,
            projection=projection,
            position=position,
            figure_kwargs=figure_kwargs,
        )

        # Plot wireframe
        if wireframe_kwargs is None:
            wireframe_kwargs = {}
        if isinstance(self.symmetry, tuple):
            fundamental_zone = OrientationRegion.from_symmetry(
                s1=self.symmetry[0], s2=self.symmetry[1]
            )
            ax.plot_wireframe(fundamental_zone, **wireframe_kwargs)
        else:
            # Orientation via inheritance
            fundamental_zone = OrientationRegion.from_symmetry(self.symmetry)
            ax.plot_wireframe(fundamental_zone, **wireframe_kwargs)

        # Correct the aspect ratio of the axes according to the extent
        # of the boundaries of the fundamental region, and also restrict
        # the data limits to these boundaries
        ax._correct_aspect_ratio(fundamental_zone, set_limits=True)

        ax.axis("off")
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)

        if size is not None:
            to_plot = self.get_random_sample(size)
        else:
            to_plot = self
        ax.scatter(to_plot, fundamental_zone=fundamental_zone, **kwargs)

        if return_figure:
            return figure

    def get_distance_matrix(
        self, chunk_size: int = 20, progressbar: bool = True
    ) -> np.ndarray:
        r"""The symmetry reduced smallest angle of rotation transforming
        every misorientation in this instance to every other
        misorientation :cite:`johnstone2020density`.

        This is an alternative implementation of
        :meth:`~orix.quaternion.Misorientation.distance` for
        a single :class:`Misorientation` instance, using :mod:`dask`.

        Parameters
        ----------
        chunk_size
            Number of misorientations per axis to include in each
            iteration of the computation. Default is 20.
        progressbar
            Whether to show a progressbar during computation. Default is
            ``True``.

        Returns
        -------
        angles
            Misorientation angles.

        Notes
        -----
        Given two misorientations :math:`m_i` and :math:`m_j` with the
        same two symmetry groups, the smallest angle is considered as
        the geodesic distance

        .. math::

            d(m_i, m_j) = \arccos(2(m_i \cdot m_j)^2 - 1),

        where :math:`(m_i \cdot m_j)` is the highest dot product
        between symmetrically equivalent misorientations to
        :math:`m_{i,j}`, given by

        .. math::

            \max_{s_k \in S_k} s_k m_i s_l s_k m_j^{-1} s_l,

        where :math:`s_k \in S_k` and :math:`s_l \in S_l`, with
        :math:`S_k` and :math:`S_l` being the two symmetry groups.

        Examples
        --------
        >>> from orix.quaternion import Misorientation, symmetry
        >>> m = Misorientation.from_axes_angles([1, 0, 0], [0, np.pi / 2])
        >>> m.symmetry = (symmetry.D6, symmetry.D6)
        >>> m.get_distance_matrix(progressbar=False)
        array([[0.        , 1.57079633],
               [1.57079633, 0.        ]])
        """
        # Reduce symmetry operations to the unique ones
        symmetry = _get_unique_symmetry_elements(*self.symmetry)

        # Perform "s_k m_i s_l s_k m_j" (see Notes)
        mori1 = symmetry.outer(self).outer(symmetry)
        mori2 = mori1._outer_dask(~self, chunk_size=chunk_size)

        # Perform last outer product and reduce to all dot products at
        # the same time
        warnings.filterwarnings("ignore", category=da.PerformanceWarning)
        str1 = "abcdefghijklmnopqrstuvwxy"[: mori2.ndim]
        str2 = "z" + str1[-1]  # Last axis has shape (4,)
        sum_over = f"{str1},{str2}->{str1[:-1] + str2[0]}"
        all_dot_products = da.einsum(sum_over, mori2, symmetry.data)

        # Get highest dot product
        axes = (0, self.ndim + 1, 2 * self.ndim + 2)
        dot_products = da.max(abs(all_dot_products), axis=axes)

        # Round because some dot products are slightly above 1
        dot_products = da.round(dot_products, 12)

        # Calculate disorientation angles
        angles_dask = da.arccos(2 * dot_products**2 - 1)
        angles_dask = da.nan_to_num(angles_dask)
        angles = np.zeros(angles_dask.shape)
        if progressbar:
            with ProgressBar():
                da.store(sources=angles_dask, targets=angles)
        else:
            da.store(sources=angles_dask, targets=angles)

        return angles

    @classmethod
    def from_align_vectors(
        cls,
        to_: Miller,
        from_: Miller,
        weights: Optional[np.ndarray] = None,
        return_rmsd: bool = False,
        return_sensitivity: bool = False,
    ) -> Misorientation:
        """ "Return an estimated misorientation to optimally align two sets of vectors, one in each crystal."

        This method wraps :meth:`scipy.spatial.transform.Rotation.align_vectors`,
        see that method for further explanations of parameters and returns.

        Parameters
        ----------
        to_
            Directions in the other crystal.
        from_
            Directions in the initial crystal.
        weights
            The relative importance of the different vectors.
        return_rmsd
            Whether to return the root mean square distance (weighted)
            between ``to_`` and ``from_`` after alignment.
        return_sensitivity
            Whether to return the sensitivity matrix.

        Returns
        -------
        estimated misorientation
            Best estimate of the ``misorientation`` that transforms ``from_`` to ``to_``.
            The symmetry of the ``misorientation`` is inferred from the phase of ``to_`` and ``from_``, if given.
        rmsd
            Returned when ``return_rmsd=True``.
        sensitivity
            Returned when ``return_sensitivity=True``.

        Examples
        -------
        >>> from orix.quaternion import Misorientation
        >>> from orix.vector import Vector3d, Miller
        >>> crystal1 = Miller([[1, 0, 0], [0, 1, 0]])
        >>> crystal2 = Miller([[1, 0, 0], [0, 0, 1]])
        >>> Misorientation.from_align_vectors(crystal1, crystal2)
        Misorientation (1,) 1
        [[-0.7071  0.7071  0.      0.    ]]
        """
        if not isinstance(to_, Miller) or not isinstance(from_, Miller):
            raise ValueError(
                "Arguments to_ and from_ must both be of type Miller, "
                f"but are of type {type(to_)} and {type(from_)}."
            )

        out = super().from_align_vectors(
            to_=to_,
            from_=from_,
            weights=weights,
            return_rmsd=return_rmsd,
            return_sensitivity=return_sensitivity,
        )
        out = list(out)
        if hasattr(to_.phase, "point_group") and hasattr(from_.phase, "point_group"):
            out[0].symmetry = (from_.phase.point_group, to_.phase.point_group)

        return out[0] if len(out) == 1 else out


class Orientation(Misorientation):
    r"""Orientations represent misorientations away from a reference of
    identity and have only one associated symmetry.

    Orientations support binary subtraction, producing a misorientation.
    That is, to compute the misorientation from :math:`g_1` to
    :math:`g_2`, call :code:`g_2 - g_1`.

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
        """Unit orientations."""
        o = super().unit
        o.symmetry = self.symmetry
        return o

    def __invert__(self) -> Orientation:
        o = super().__invert__()
        o.symmetry = self.symmetry
        return o

    def __neg__(self) -> Orientation:
        o = super().__neg__()
        o.symmetry = self.symmetry
        return o

    def __repr__(self) -> str:
        """String representation."""
        data = np.array_str(self.data, precision=4, suppress_small=True)
        return f"{self.__class__.__name__} {self.shape} {self.symmetry.name}\n{data}"

    def __sub__(self, other: Orientation) -> Misorientation:
        if isinstance(other, Orientation):
            # Call to Object3d.squeeze() doesn't carry over symmetry
            misorientation = Misorientation(self * ~other).squeeze()
            misorientation.symmetry = (self.symmetry, other.symmetry)
            return misorientation.map_into_symmetry_reduced_zone()
        return NotImplemented

    # TODO: Remove use of **kwargs in 1.0
    @classmethod
    def from_euler(
        cls,
        euler: np.ndarray,
        symmetry: Optional[Symmetry] = None,
        direction: str = "lab2crystal",
        **kwargs,
    ) -> Orientation:
        """Return orientation(s) from an array of Euler angles.

        Parameters
        ----------
        euler
            Euler angles in the Bunge convention.
        symmetry
            Symmetry of orientation(s). If not given (default), no
            symmetry is set.
        direction
            ``"lab2crystal"`` (default) or ``"crystal2lab"``.
            ``"lab2crystal"`` is the Bunge convention. If ``"MTEX"`` is
            provided then the direction is ``"crystal2lab"``.

        Returns
        -------
        ori
            Orientations.
        """
        ori = super().from_euler(euler=euler, direction=direction, **kwargs)
        if symmetry:
            ori.symmetry = symmetry
        return ori

    @classmethod
    def from_align_vectors(
        cls,
        to_: Miller,
        from_: Vector3d,
        weights: Optional[np.ndarray] = None,
        return_rmsd: bool = False,
        return_sensitivity: bool = False,
    ) -> Orientation:
        """Return an estimated orientation to optimally align vectors in the crystal and sample reference frames.

        This method wraps :meth:`scipy.spatial.transform.Rotation.align_vectors`,
        see that method for further explanations of parameters and returns.

        Parameters
        ----------
        to_
            Directions in the other crystal.
        from_
            Directions in the initial crystal.
        weights
            The relative importance of the different vectors.
        return_rmsd
            Whether to return the root mean square distance (weighted)
            between ``to_`` and ``from_`` after alignment.
        return_sensitivity
            Whether to return the sensitivity matrix.

        Returns
        -------
        estimated orientation
            Best estimate of the ``orientation`` that transforms ``from_`` to ``to_``.
            The symmetry of the ``orientaiton`` is inferred form the point group of the phase of ``to_``, if given.
        rmsd
            Returned when ``return_rmsd=True``.
        sensitivity
            Returned when ``return_sensitivity=True``.

        Examples
        -------
        >>> from orix.quaternion import Orientation
        >>> from orix.vector import Vector3d, Miller
        >>> crystal_millers = Miller([[2, -1 ,0], [0, 0, 1]])
        >>> sample_vectors = Vector3d([[3, 1, 0], [-1, 3, 0]])
        >>> ori = Orientation.from_align_vectors(crystal_millers, sample_vectors)
        >>> np.allclose(crystal_millers.data, (ori * sample_vectors.unit * crystal_millers.length).data)
        True
        """
        if not isinstance(to_, Miller):
            raise ValueError(
                f"Argument to_ must be of type Miller, but has type {type(to_)}"
            )

        out = Rotation.from_align_vectors(
            to_=to_,
            from_=from_,
            weights=weights,
            return_rmsd=return_rmsd,
            return_sensitivity=return_sensitivity,
        )
        out = list(out)
        out[0] = cls(out[0].data)

        if hasattr(to_.phase, "point_group"):
            out[0].symmetry = to_.phase.point_group

        return out[0] if len(out) == 1 else out

    @classmethod
    def from_matrix(
        cls, matrix: np.ndarray, symmetry: Optional[Symmetry] = None
    ) -> Orientation:
        """Return orientation(s) from orientation matrices
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
        ori
            Orientations.
        """
        ori = super().from_matrix(matrix)
        if symmetry:
            ori.symmetry = symmetry
        return ori

    @classmethod
    def from_neo_euler(
        cls, neo_euler: NeoEuler, symmetry: Optional[Symmetry] = None
    ) -> Orientation:
        """Return orientation(s) from a neo-euler (vector)
        representation.

        Parameters
        ----------
        neo_euler
            Vector parametrization of orientation(s).
        symmetry
            Symmetry of orientation(s). If not given (default), no
            symmetry is set.

        Returns
        -------
        ori
            Orientations.
        """
        ori = super().from_neo_euler(neo_euler)
        if symmetry:
            ori.symmetry = symmetry
        return ori

    @classmethod
    def from_axes_angles(
        cls,
        axes: Union[np.ndarray, Vector3d, tuple, list],
        angles: Union[np.ndarray, tuple, list],
        symmetry: Optional[Symmetry] = None,
    ) -> Orientation:
        """Return orientations from axis-angle pairs.

        Parameters
        ----------
        axes
            The axes of rotation.
        angles
            The angles of rotation, in radians.
        symmetry
            Symmetry of orientations. If not given (default), no
            symmetry is set.

        Returns
        -------
        ori
            Orientations.

        See Also
        --------
        from_neo_euler

        Examples
        --------
        >>> from orix.quaternion import Orientation, symmetry
        >>> ori = Orientation.from_axes_angles((0, 0, -1), np.pi / 2, symmetry.Oh)
        >>> ori
        Orientation (1,) m-3m
        [[ 0.7071  0.      0.     -0.7071]]
        """
        axangle = AxAngle.from_axes_angles(axes, angles)
        return cls.from_neo_euler(axangle, symmetry)

    def angle_with(self, other: Orientation) -> np.ndarray:
        """Return the smallest symmetry reduced angles of rotation
        transforming the orientations to the other orientations.

        Parameters
        ----------
        other
            Another orientation.

        Returns
        -------
        angles
            Smallest symmetry reduced angles.

        See Also
        --------
        angle_with_outer
        Rotation.angle_with_outer
        """
        dot_products = self.unit.dot(other.unit)
        angles = np.nan_to_num(np.arccos(2 * dot_products**2 - 1))
        return angles

    def angle_with_outer(
        self,
        other: Orientation,
        lazy: bool = False,
        chunk_size: int = 20,
        progressbar: bool = True,
    ) -> np.ndarray:
        r"""The symmetry reduced smallest angle of rotation transforming
        every orientation in this instance to every orientation in
        another instance.

        This is an alternative implementation of
        :meth:`~orix.quaternion.Misorientation.distance` for
        a single :class:`Orientation` instance, using :mod:`dask`.

        Parameters
        ----------
        other
            Another orientation.
        lazy
            Whether to perform the computation lazily with Dask. Default
            is False.
        chunk_size
            Number of orientations per axis to include in each iteration
            of the computation. Default is 20. Only applies when
            ``lazy=True``.
        progressbar
            Whether to show a progressbar during computation if
            ``lazy=True``. Default is ``True``.

        Returns
        -------
        angles
            Smallest symmetry reduced angles.

        See Also
        --------
        angle_with

        Notes
        -----
        Given two orientations :math:`g_i` and :math:`g_j`, the smallest
        angle is considered as the geodesic distance

        .. math::

            d(g_i, g_j) = \arccos(2(g_i \cdot g_j)^2 - 1),

        where :math:`(g_i \cdot g_j)` is the highest dot product between
        symmetrically equivalent orientations to :math:`g_{i,j}`.

        Examples
        --------
        >>> from orix.quaternion import Orientation, symmetry
        >>> ori1 = Orientation.random((5, 3))
        >>> ori2 = Orientation.random((6, 2))
        >>> dist1 = ori1.angle_with_outer(ori2)
        >>> dist1.shape
        (5, 3, 6, 2)
        >>> ori1.symmetry = symmetry.Oh
        >>> ori2.symmetry = symmetry.Oh
        >>> dist_sym = ori1.angle_with_outer(ori2)
        >>> np.allclose(dist1.data, dist_sym.data)
        False
        """
        ori = self.unit
        if lazy:
            dot_products = ori._dot_outer_dask(other, chunk_size=chunk_size)
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
            dot_products = ori.dot_outer(other)
            angles = np.arccos(2 * dot_products**2 - 1)
            angles = np.nan_to_num(angles)

        return angles

    def get_distance_matrix(
        self, lazy: bool = False, chunk_size: int = 20, progressbar: bool = True
    ) -> np.ndarray:
        r"""Return the symmetry reduced smallest angle of rotation
        transforming all these orientations to all the other
        orientations :cite:`johnstone2020density`.

        This is an alternative implementation of
        :meth:`~orix.quaternion.Misorientation.distance` for
        a single :class:`Orientation` instance, using :mod:`dask`.

        Parameters
        ----------
        lazy
            Whether to perform the computation lazily with Dask. Default
            is ``False``.
        chunk_size
            Number of orientations per axis to include in each iteration
            of the computation. Default is 20. Only applies when
            ``lazy=True``.
        progressbar
            Whether to show a progressbar during computation if
            ``lazy=True``. Default is ``True``.

        Returns
        -------
        angles
            Symmetry reduced angles.

        Notes
        -----
        Given two orientations :math:`g_i` and :math:`g_j`, the smallest
        angle is considered as the geodesic distance

        .. math::

            d(g_i, g_j) = \arccos(2(g_i \cdot g_j)^2 - 1),

        where :math:`(g_i \cdot g_j)` is the highest dot product between
        symmetrically equivalent orientations to :math:`g_{i,j}`.
        """
        angles = self.angle_with_outer(
            self, lazy=lazy, chunk_size=chunk_size, progressbar=progressbar
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
        >>> ori1 = Orientation.from_axes_angles([0, 0, 1], np.deg2rad([0, 45]), symmetry.Oh)
        >>> ori2 = Orientation.from_axes_angles([0, 0, 1], np.deg2rad([45, 90]), symmetry.Oh)
        >>> ori1.dot(ori2)
        array([0.92387953, 0.92387953])
        """
        symmetry = _get_unique_symmetry_elements(self.symmetry, other.symmetry)
        misorientation = other * ~self
        all_dot_products = Rotation(misorientation).dot_outer(symmetry)
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
        >>> ori1 = Orientation.from_axes_angles([0, 0, 1], np.deg2rad([0, 45]), symmetry.Oh)
        >>> ori2 = Orientation.from_axes_angles([0, 0, 1], np.deg2rad([45, 90]), symmetry.Oh)
        >>> ori1.dot_outer(ori2)
        array([[0.92387953, 1.        ],
               [1.        , 0.92387953]])
        """
        symmetry = _get_unique_symmetry_elements(self.symmetry, other.symmetry)
        misorientation = other.outer(~self)
        all_dot_products = Rotation(misorientation).dot_outer(symmetry)
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
        ori = pg._special_rotation.outer(self)

        alpha, beta, gamma = ori.to_euler().T
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
        position: Union[int, Tuple[int], SubplotSpec] = None,
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

            figure.tight_layout()

        if return_figure:
            return figure

    def _dot_outer_dask(self, other: Orientation, chunk_size: int = 20) -> da.Array:
        """Symmetry reduced dot product of every orientation in this
        instance to every orientation in another instance, returned as a
        Dask array.

        Parameters
        ----------
        other
        chunk_size
            Number of orientations per axis in each orientation instance
            to include in each iteration of the computation. Default is
            20.

        Returns
        -------
        highest_dot_product

        Notes
        -----
        To read the dot products array `dparr` into memory, do
        `dp = dparr.compute()`.
        """
        symmetry = _get_unique_symmetry_elements(self.symmetry, other.symmetry)
        misorientation = other._outer_dask(~self, chunk_size=chunk_size)

        # Summation subscripts
        str1 = "abcdefghijklmnopqrstuvwxy"[: misorientation.ndim]
        str2 = "z" + str1[-1]  # Last elements have shape (4,)
        sum_over = f"{str1},{str2}->{str1[:-1] + str2[0]}"

        warnings.filterwarnings("ignore", category=da.PerformanceWarning)

        all_dot_products = da.einsum(sum_over, misorientation, symmetry.data)
        highest_dot_product = da.max(abs(all_dot_products), axis=-1)

        return highest_dot_product
