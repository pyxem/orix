# -*- coding: utf-8 -*-
# Copyright 2018-2023 the orix developers
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
from matplotlib.gridspec import SubplotSpec
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation
from tqdm import tqdm

from orix.quaternion.orientation_region import OrientationRegion
from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C1, Symmetry, _get_unique_symmetry_elements
from orix.vector import Miller


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

    def __init__(
        self,
        data: Union[np.ndarray, Misorientation, list, tuple],
        symmetry: Optional[Tuple[Symmetry, Symmetry]] = None,
    ):
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
            misorientations = Misorientation.stack([self, ~self]).flatten()
        else:
            misorientations = Misorientation(self)

        equivalent = Gr.outer(misorientations.outer(Gl))
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
        ax._correct_aspect_ratio(fundamental_zone)

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
        self,
        chunk_size: int = 20,
        progressbar: bool = True,
        degrees: bool = False,
    ) -> np.ndarray:
        r"""Return the symmetry reduced smallest angle of rotation
        transforming every misorientation in this instance to every
        other misorientation :cite:`johnstone2020density`.

        Parameters
        ----------
        chunk_size
            Number of misorientations per axis to include in each
            iteration of the computation. Default is 20. Increasing this
            might reduce the computation time at the cost of increased
            memory use.
        progressbar
            Whether to show a progressbar during computation. Default is
            ``True``.
        degrees
            If ``True``, the angles are returned in degrees. Default is
            ``False``.

        Returns
        -------
        angles
            Misorientation angles in radians (``degrees=False``) or
            degrees (``degrees=True``).

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
        >>> m = Misorientation.from_axes_angles([1, 0, 0], [0, 90], degrees=True)
        >>> m.symmetry = (symmetry.D6, symmetry.D6)
        >>> m.get_distance_matrix(progressbar=False, degrees=True)
        array([[ 0., 90.],
               [90.,  0.]])
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

        if degrees:
            angles = np.rad2deg(angles)

        return angles

    @classmethod
    def from_align_vectors(
        cls,
        other: Miller,
        initial: Miller,
        weights: Optional[np.ndarray] = None,
        return_rmsd: bool = False,
        return_sensitivity: bool = False,
    ) -> Union[
        Misorientation,
        Tuple[Misorientation, float],
        Tuple[Misorientation, np.ndarray],
        Tuple[Misorientation, float, np.ndarray],
    ]:
        """Return an estimated misorientation to optimally align two
        sets of vectors, one set in each crystal.

        This method wraps
        :meth:`~scipy.spatial.transform.Rotation.align_vectors`. See
        that method for further explanations of parameters and returns.

        Parameters
        ----------
        other
            Directions of shape ``(n,)`` in the other crystal.
        initial
            Directions of shape ``(n,)`` in the initial crystal.
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
        estimated_misorientation
            Best estimate of the misorientation that transforms
            ``initial`` to ``other``. The symmetry of the misorientation
            is inferred from the phase of ``other`` and ``initial``, if
            given.
        rmsd
            Returned when ``return_rmsd=True``.
        sensitivity
            Returned when ``return_sensitivity=True``.

        Raises
        ------
        ValueError
            If ``other`` and ``initial`` are not Miller instances.

        Examples
        --------
        >>> from orix.quaternion import Misorientation
        >>> from orix.vector import Miller
        >>> from orix.crystal_map import Phase
        >>> uvw1 = Miller(uvw=[[1, 0, 0], [0, 1, 0]], phase=Phase(point_group="m-3m"))
        >>> uvw2 = Miller(uvw=[[1, 0, 0], [0, 0, 1]], phase=Phase(point_group="m-3m"))
        >>> mori12 = Misorientation.from_align_vectors(uvw2, uvw1)
        >>> mori12 * uvw1
        Miller (2,), point group m-3m, uvw
        [[1. 0. 0.]
         [0. 0. 1.]]
        """
        if not isinstance(other, Miller) or not isinstance(initial, Miller):
            raise ValueError(
                "Arguments other and initial must both be of type Miller, "
                f"but are of type {type(other)} and {type(initial)}."
            )

        out = super().from_align_vectors(
            other=other,
            initial=initial,
            weights=weights,
            return_rmsd=return_rmsd,
            return_sensitivity=return_sensitivity,
        )
        out = list(out)

        try:
            out[0].symmetry = (initial.phase.point_group, other.phase.point_group)
        except (AttributeError, ValueError):
            pass

        return out[0] if len(out) == 1 else tuple(out)

    @classmethod
    def from_scipy_rotation(
        cls,
        rotation: SciPyRotation,
        symmetry: Optional[Tuple[Symmetry, Symmetry]] = None,
    ) -> Misorientation:
        """Return misorientations(s) from
        :class:`scipy.spatial.transform.Rotation`.

        Parameters
        ----------
        rotation
            SciPy rotation(s).
        symmetry
            Tuple of two sets of crystal symmetries. If not given, the
            returned misorientation(s) is assumed to be transformation
            between crystals with only the identity operation, *1*
            (*C1*).

        Returns
        -------
        misorientation
            Misorientation(s).

        Notes
        -----
        The SciPy rotation is inverted to be consistent with the orix
        framework of passive rotations.

        Examples
        --------
        >>> from orix.crystal_map import Phase
        >>> from orix.quaternion import Misorientation, symmetry
        >>> from orix.vector import Miller
        >>> from scipy.spatial.transform import Rotation as SciPyRotation
        >>> r_scipy = SciPyRotation.from_euler("ZXZ", [90, 0, 0], degrees=True)
        >>> mori = Misorientation.from_scipy_rotation(
        ...     r_scipy, (symmetry.Oh, symmetry.Oh)
        ... )
        >>> uvw = Miller(uvw=[1, 1, 0], phase=Phase(point_group="m-3m"))
        >>> r_scipy.apply(uvw.data)
        array([[-1.,  1.,  0.]])
        >>> mori * uvw
        Miller (1,), point group m-3m, uvw
        [[ 1. -1.  0.]]
        >>> ~mori * uvw
        Miller (1,), point group m-3m, uvw
        [[-1.  1.  0.]]
        """
        mori = super().from_scipy_rotation(rotation)
        if symmetry:
            mori.symmetry = symmetry
        return mori
