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

from itertools import product as iproduct
from typing import Any, List, Optional, Tuple, Union
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
    r"""Misorientations :math:`M`.

    Misorientations represent transformations from one orientation,
    :math:`O_1` to another, :math:`O_2`: :math:`O_2 \cdot O_1^{-1}`.

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

    # -------------------------- Properties -------------------------- #

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

    # ------------------------ Dunder methods ------------------------ #

    def __eq__(self, other: Union[Any, Misorientation]) -> bool:
        v1 = super().__eq__(other)
        if not v1:
            return v1
        else:
            # Check whether symmetries also are equivalent
            v2 = []
            for sym_s, sym_o in zip(self._symmetry, other._symmetry):
                v2.append(sym_s == sym_o)
            return all(v2)

    def __getitem__(self, key) -> Misorientation:
        M = super().__getitem__(key)
        M._symmetry = self._symmetry
        return M

    def __invert__(self) -> Misorientation:
        M = super().__invert__()
        M._symmetry = self._symmetry[::-1]
        return M

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

    # ------------------------ Class methods ------------------------- #

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
        >>> t1 = Miller(uvw=[[1, 0, 0], [0, 1, 0]], phase=Phase(point_group="m-3m"))
        >>> t2 = Miller(uvw=[[1, 0, 0], [0, 0, 1]], phase=Phase(point_group="m-3m"))
        >>> M12 = Misorientation.from_align_vectors(t2, t1)
        >>> M12 * t1
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
        """Return misorientationss from
        :class:`scipy.spatial.transform.Rotation`.

        Parameters
        ----------
        rotation
            SciPy rotations.
        symmetry
            Tuple of two sets of crystal symmetries. If not given, the
            returned misorientations are assumed to be transformations
            between crystals with only the identity operation, *1*
            (*C1*).

        Returns
        -------
        M
            Misorientations.

        Notes
        -----
        The SciPy rotations are inverted to be consistent with the orix
        framework of rotations.

        Examples
        --------
        >>> from orix.crystal_map import Phase
        >>> from orix.quaternion import Misorientation, symmetry
        >>> from orix.vector import Miller
        >>> from scipy.spatial.transform import Rotation as SciPyRotation
        >>> R_scipy = SciPyRotation.from_euler("ZXZ", [90, 0, 0], degrees=True)
        >>> M = Misorientation.from_scipy_rotation(
        ...     R_scipy, (symmetry.Oh, symmetry.Oh)
        ... )
        >>> t = Miller(uvw=[1, 1, 0], phase=Phase(point_group="m-3m"))
        >>> R_scipy.apply(t.data)
        array([[-1.,  1.,  0.]])
        >>> M * t
        Miller (1,), point group m-3m, uvw
        [[ 1. -1.  0.]]
        >>> ~M * t
        Miller (1,), point group m-3m, uvw
        [[-1.  1.  0.]]
        """
        M = super().from_scipy_rotation(rotation)
        if symmetry:
            M.symmetry = symmetry
        return M

    @classmethod
    def random(
        cls,
        shape: Union[int, tuple] = 1,
        symmetry: Optional[Tuple[Symmetry, Symmetry]] = None,
    ) -> Misorientation:
        """Create random misorientations.

        Parameters
        ----------
        shape
            Shape of the misorientations.
        symmetry
            Tuple of two sets of crystal symmetries. If not given, the
            returned misorientation(s) is assumed to be transformation
            between crystals with only the identity operation, *1*
            (*C1*).

        Returns
        -------
        M
            Random misorientations.
        """
        M = super().random(shape)
        if symmetry:
            M.symmetry = symmetry
        return M

    # --------------------- Other public methods --------------------- #

    def reshape(self, *shape) -> Misorientation:
        M = super().reshape(*shape)
        M._symmetry = self._symmetry
        return M

    def flatten(self) -> Misorientation:
        M = super().flatten()
        M._symmetry = self._symmetry
        return M

    def squeeze(self) -> Misorientation:
        M = super().squeeze()
        M._symmetry = self._symmetry
        return M

    def transpose(self, *axes) -> Misorientation:
        M = super().transpose(*axes)
        M._symmetry = self._symmetry
        return M

    def equivalent(self, grain_exchange: bool = False) -> Misorientation:
        r"""Return the equivalent misorientations.

        Parameters
        ----------
        grain_exchange
            If ``True`` the rotation :math:`g` and :math:`g^{-1}` are
            considered to be identical. Default is ``False``.

        Returns
        -------
        M
            The equivalent misorientations.
        """
        Gl, Gr = self._symmetry

        if grain_exchange and (Gl._tuples == Gr._tuples):
            M = Misorientation.stack((self, ~self)).flatten()
        else:
            M = Misorientation(self)

        equivalent = Gr.outer(M.outer(Gl))

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
        M
            A new misorientation object with the assigned symmetry.

        Examples
        --------
        >>> from orix.quaternion.symmetry import C4, C2
        >>> data = np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 0, 0]])
        >>> M = Misorientation(data)
        >>> M.symmetry = (C4, C2)
        >>> M.map_into_symmetry_reduced_zone()
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

    def scatter(
        self,
        projection: str = "axangle",
        figure: Optional[plt.Figure] = None,
        position: Union[int, Tuple[int, int], SubplotSpec] = (1, 1, 1),
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
        Given two misorientations :math:`M_i` and :math:`M_j` with the
        same two symmetry groups, the smallest angle is considered as
        the geodesic distance

        .. math::

            d(M_i, M_j) = \arccos(2(M_i \cdot M_j)^2 - 1),

        where :math:`(M_i \cdot M_j)` is the highest dot product
        between symmetrically equivalent misorientations to
        :math:`M_{i,j}`, given by

        .. math::

            \max_{s_k \in S_k} s_k M_i s_l s_k M_j^{-1} s_l,

        where :math:`s_k \in S_k` and :math:`s_l \in S_l`, with
        :math:`S_k` and :math:`S_l` being the two symmetry groups.

        Examples
        --------
        >>> from orix.quaternion import Misorientation, symmetry
        >>> M = Misorientation.from_axes_angles([1, 0, 0], [0, 90], degrees=True)
        >>> M.symmetry = (symmetry.D6, symmetry.D6)
        >>> M.get_distance_matrix(progressbar=False, degrees=True)
        array([[ 0., 90.],
               [90.,  0.]])
        """
        # Reduce symmetry operations to the unique ones
        symmetry = _get_unique_symmetry_elements(*self.symmetry)

        # Perform "s_k m_i s_l s_k m_j" (see Notes)
        M1 = symmetry.outer(self).outer(symmetry)
        M2 = M1._outer_dask(~self, chunk_size=chunk_size)

        # Perform last outer product and reduce to all dot products at
        # the same time
        warnings.filterwarnings("ignore", category=da.PerformanceWarning)
        str1 = "abcdefghijklmnopqrstuvwxy"[: M2.ndim]
        str2 = "z" + str1[-1]  # Last axis has shape (4,)
        sum_over = f"{str1},{str2}->{str1[:-1] + str2[0]}"
        all_dot_products = da.einsum(sum_over, M2, symmetry.data)

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

    def inv(self) -> Misorientation:
        r"""Return the inverse misorientations :math:`M^{-1}`."""
        return self.__invert__()
