#
# Copyright 2018-2026 the orix developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix. If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import annotations

from itertools import product as iproduct
import logging
import sys
from typing import Any, Literal, overload
import warnings

import dask.array as da
from dask.diagnostics.progress import ProgressBar
import matplotlib.figure as mfigure
from matplotlib.gridspec import SubplotSpec
import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation
from tqdm import tqdm

from orix.quaternion._numba import _mori_distance_matrix
from orix.quaternion.orientation_region import OrientationRegion
from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import (
    C1,
    Symmetry,
    _get_unique_symmetry_elements,
)
from orix.vector.miller import Miller

_logger = logging.getLogger(__name__)


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
        data: np.ndarray | Misorientation | list | tuple,
        symmetry: tuple[Symmetry, Symmetry] | None = None,
    ) -> None:
        super().__init__(data)
        if symmetry:
            self.symmetry = symmetry

    # -------------------------- Properties -------------------------- #

    @property
    def symmetry(self) -> tuple[Symmetry, Symmetry]:
        """Return or set the crystal symmetries.

        Parameters
        ----------
        value : list of Symmetry or 2-tuple of Symmetry
            Crystal symmetries.
        """
        return self._symmetry

    @symmetry.setter
    def symmetry(self, value: list[Symmetry] | tuple[Symmetry, Symmetry]) -> None:
        if not isinstance(value, (list, tuple)):
            raise TypeError("Value must be a 2-tuple of Symmetry objects.")
        if len(value) != 2 or not all(isinstance(s, Symmetry) for s in value):
            raise ValueError("Value must be a 2-tuple of Symmetry objects.")
        self._symmetry = tuple(value)

    # ------------------------ Dunder methods ------------------------ #

    def __eq__(self, other: Any | Misorientation) -> bool:
        v1 = super().__eq__(other)
        if not v1:
            return v1
        else:
            # Check whether symmetries also are equivalent
            v2 = []
            for sym_s, sym_o in zip(self._symmetry, other._symmetry):
                v2.append(sym_s == sym_o)
            return all(v2)

    def __getitem__(self, key: Any) -> Misorientation:
        M = super().__getitem__(key)
        M._symmetry = self._symmetry
        return M

    def __invert__(self) -> Misorientation:
        M = super().__invert__()
        M._symmetry = self._symmetry[::-1]
        return M

    def __repr__(self) -> str:
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
        weights: np.ndarray | None = None,
        return_rmsd: bool = False,
        return_sensitivity: bool = False,
    ) -> (
        Misorientation
        | tuple[Misorientation, float]
        | tuple[Misorientation, np.ndarray]
        | tuple[Misorientation, float, np.ndarray]
    ):
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
            out[0].symmetry = (
                initial.phase.point_group,
                other.phase.point_group,
            )
        except (AttributeError, ValueError):
            pass

        return out[0] if len(out) == 1 else tuple(out)

    @classmethod
    def from_scipy_rotation(
        cls,
        rotation: SciPyRotation,
        symmetry: tuple[Symmetry, Symmetry] | None = None,
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
    def from_path_ends(
        cls, points: Misorientation, closed: bool = False, steps: int = 100
    ) -> Misorientation:
        """Return misorientations tracing the shortest path between two
        or more consecutive points.

        Parameters
        ----------
        points
            Two or more misorientations that define points along the
            path.
        closed
            Add a final trip from the last point back to the first, thus
            closing the loop. Default is False.
        steps
            Number of misorientations to return between each point along
            the path given by *points*. Default is 100.

        Returns
        -------
        path
            Regularly spaced misorientations along the path.

        See Also
        --------
        :class:`~orix.quaternion.Quaternion.from_path_ends`,
        :class:`~orix.quaternion.Orientation.from_path_ends`

        Notes
        -----
        This function traces the shortest path between points without
        considering symmetry. The concept of "shortest path" is not
        well-defined for misorientations, which can define multiple
        symmetrically equivalent points with non-equivalent paths.
        """
        points_type = type(points)
        if points_type is not cls:
            raise TypeError(
                f"Points must be misorientations, not of type {points_type}"
            )
        out = Rotation.from_path_ends(points=points, closed=closed, steps=steps)
        return cls(out.data, symmetry=points.symmetry)

    @classmethod
    def random(
        cls,
        shape: int | tuple = 1,
        symmetry: tuple[Symmetry, Symmetry] | None = None,
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

    def reshape(self, *shape: tuple[int, ...]) -> Misorientation:
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

    def transpose(self, *axes: tuple[int, ...]) -> Misorientation:
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

    def reduce(self, verbose: bool = False) -> Misorientation:
        """Return symmetrically equivalent transformations with the
        smallest angle of rotation.

        For misorientations, reduced representations are further
        restricted to transforms inside their symmetry's fundamental
        zone. See *Notes* for details.

        Parameters
        ----------
        verbose
            Whether to print a progressbar. Default is False.

        Returns
        -------
        mori
            New misorientation with the assigned symmetry.

        See Also
        --------
        :class:`~orix.quaternion.OrientationRegion`

        Notes
        -----
        In orix, a (mis)orientation fundamental zone (FZ) is defined as
        a bounded volume in quaternion space containing only proper
        rotations. This is a common convention (see
        :meth:`~orix.quaternion.OrientationRegion.from_symmetry` for
        details). This is relevant for defining a reduced representation
        because the brute force expansion of a misorientation to its
        symmetric equivalents can produce up to four unique rotations
        that fall inside the FZ and also all have the same rotation
        angle.

        For all orientations and 924/1024 possible symmetry combinations
        (32^2) for misorientations, this fact is irrelevant, as either
        only a single proper rotation or an identical proper and
        improper pair will map to the FZ. The remaining 100 cases occur
        when both symmetries are improper and do not possess an
        inversion. In this case, four unique values fall within the FZ,
        including two proper rotations. One of these is produced using
        only proper rotations, whereas the second is a 'pseudo-proper'
        rotation resulting from two consecutive rotoinversions. Since
        pseudo-proper variants cannot be reached through any combination
        of proper rotations from either symmetry, they are ignored by
        orix and only the proper rotation is returned.

        If *verbose* is True, the progress bar will most likely
        terminate before trying all iterations, as all (mis)orientations
        are within the fundamental zone already.

        Examples
        --------
        >>> from orix.quaternion import Misorientation
        >>> from orix.quaternion.symmetry import C4, C2
        >>> mori = Misorientation([[0.5, 0.5, 0.5, 0.5], [0, 1, 0, 0]])
        >>> mori.symmetry = (C4, C2)
        >>> mori.reduce()
        Misorientation (2,) 4, 2
        [[-0.7071 0.     -0.7071  0.    ]
        [ 0.      1.      0.      0.    ]]
        """
        # Combine symmetry elements of start and end of transformation
        # given by the (mis)orientation
        start, end = self._symmetry
        symmetry_pairs = iproduct(start, end)

        log_msg = "Reducing to fundamental zone"
        if verbose:
            symmetry_pairs = tqdm(
                symmetry_pairs, total=start.size * end.size, desc=log_msg
            )
        else:
            _logger.info(log_msg)

        # Find the (mis)orientations which lie inside the Rodrigues
        # (orientation) or MacKenzie (misorientation) fundamental zone
        # (FZ), given by the symmetry elements. We loop over all
        # symmetry pairs and rotate all (mis)orientations until all are
        # inside the FZ. Ignore symmetry combinations that need an
        # inversion, as these are not handled by the MacKenzie/Rodrigues
        # definitions of a FZ.
        fz = OrientationRegion.from_symmetry(start=start, end=end)
        reduced = self.__class__.identity(self.shape)
        is_outside = np.ones(self.shape, dtype=bool)
        for sym_start, sym_end in symmetry_pairs:
            if sym_start.improper != sym_end.improper:
                continue
            reduced[is_outside] = sym_end * self[is_outside] * sym_start
            is_outside = ~(reduced < fz)
            if not is_outside.any():
                break

        # Convert to northern hemisphere representations
        reduced.data[reduced.a < 0] = reduced.data[reduced.a < 0] * -1

        reduced._symmetry = (start, end)

        return reduced

    def scatter(
        self,
        projection: Literal["axangle", "rodrigues", "homochoric"] = "axangle",
        figure: mfigure.Figure | None = None,
        position: int | tuple[int, int, int] | SubplotSpec = (1, 1, 1),
        return_figure: bool = False,
        wireframe_kwargs: dict | None = None,
        size: int | None = None,
        figure_kwargs: dict | None = None,
        **kwargs,
    ) -> mfigure.Figure | None:
        """Plot misorientations in 3D Euclidean space using a
        Neo-Eulerian projection.

        Parameters
        ----------
        projection
            Which axis-angle projection to use for plotting into
            Euclidean space. The options are "axangle" (default) for a
            linear scaling, "homochoric" for an equal-volume scaling, or
            "rodrigues" for a rectilinear scaling.
        figure
            If given, a new plot axis with the projection specified by
            *projection* is added to the figure in the position
            specified by *position*. If not given, a new figure is
            created.
        position
            Where to add the new plot axis. 121 or (1, 2, 1) places it
            in the first of two positions in a grid of 1 row and 2
            columns. See :meth:`~matplotlib.figure.Figure.add_subplot`
            for further details. Default is (1, 1, 1).
        return_figure
            Whether to return the figure. Default is False.
        wireframe_kwargs
            Keyword arguments passed to
            :meth:`~orix.plot.AxAnglePlot.plot_wireframe` or equivalent.
        size
            If not given, all misorientations are plotted. If given, a
            random sample of this size of the misorientations is
            plotted.
        figure_kwargs
            Dictionary of keyword arguments passed to
            :func:`matplotlib.pyplot.figure` if *figure* is not given.
        **kwargs
            Keyword arguments passed to the orix plotting class set by
            *position*.

        Returns
        -------
        figure
            Figure with the added plot axis, if *return_figure* is True.

        See Also
        --------
        orix.quaternion.Orientation.scatter
        :meth:`~orix.plot.AxAnglePlot`
        :meth:`~orix.plot.RodriguesPlot`
        :meth:`~orix.plot.HomochoricPlot`
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
        *,
        lazy: bool = True,
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
            memory use. Only used if *lazy* is True.
        progressbar
            Whether to show a progressbar during computation. Default is
            True. Only used if *lazy* is True.
        degrees
            If True, the angles are returned in degrees. Default is
            False.
        lazy
            Whether to compute with Dask. Default is True. Setting False
            should be both faster and use less memory.

            False will be the default in the future.

        Returns
        -------
        angles
            Misorientation angles in radians (*degrees* is False) or
            degrees (*degrees* is True).

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
        >>> mori = Misorientation.from_axes_angles([1, 0, 0], [0, 90], degrees=True)
        >>> mori.symmetry = (symmetry.D6, symmetry.D6)
        >>> mori.get_distance_matrix(progressbar=False, degrees=True)
        array([[ 0., 90.],
               [90.,  0.]])
        """
        # Reduce symmetry operations to the unique ones
        symmetry = _get_unique_symmetry_elements(*self.symmetry)

        if lazy:
            angles = _get_distance_matrix_dask(
                mori=self,
                symmetry=symmetry,
                chunk_size=chunk_size,
                progressbar=progressbar,
            )
        else:
            qu = np.ascontiguousarray(self.unit.data.reshape(-1, 4), dtype=np.float64)
            sym_ops = np.ascontiguousarray(symmetry.data, dtype=np.float64)
            angles = _mori_distance_matrix(qu, sym_ops)
            angles = angles.reshape(self.shape + self.shape)

        if degrees:
            angles = np.rad2deg(angles)

        return angles

    def inv(self) -> Misorientation:
        r"""Return the inverse misorientations :math:`M^{-1}`."""
        return self.__invert__()

    @overload
    def mean(
        self,
        weights: np.ndarray | None = None,
        include_improper: bool = False,
        use_symmetry: bool = True,
        return_neighbors: Literal[False] = False,
        verbose: bool = False,
    ) -> Misorientation: ...

    @overload
    def mean(
        self,
        weights: np.ndarray | None = None,
        include_improper: bool = False,
        use_symmetry: bool = True,
        return_neighbors: Literal[True] = True,
        verbose: bool = False,
    ) -> tuple[Misorientation, Rotation]: ...

    def mean(
        self,
        weights: np.ndarray | None = None,
        include_improper: bool = False,
        use_symmetry: bool = True,
        return_neighbors: bool = False,
        verbose: bool = False,
    ) -> Misorientation | tuple[Misorientation, Rotation]:
        """Return the symmetry-respecting mean (mis)orientation.

        Parameters
        ----------
        weights
            Weights for calculating a weighted average. Must be of the
            same size as the (mis)orientation, or None.
        include_improper
            Whether to exclude equivalent representations that require
            inversion symmetry to calculate. Default is False.
        use_symmetry
            Whether to consider symmetry. Default is True. If False, the
            calculation is the same as for rotations.
        return_neighbors
            Whether to return the nearest neighbors used to calculate
            the mean. Default is False.
        verbose
            Whether to print progressbars. Default is False.

        Returns
        -------
        fine_mean
            Mean (mis)orientation.
        neighbors
            If *return_neighbors* is True, returns the representations
            used to calculate the mean.

        Notes
        -----
        This method uses the Frobenius norm of rotation space to define
        a mean for rotations, as given in equations 12 and 13 of
        :cite:`markley_averaging_2007`. Refer to
        :meth:`~orix.quaternion.Quaternion.mean` for details.

        To account for symmetry, the following procedure is used:

        1. Misorientations are reduced to the fundamental zone.
        2. A rough mean is calculated.
        3. Misorientations with equivalent values closer to the rough
           mean are updated to the nearby value.
        4. The precise mean is recalculated.

        If *use_symmetry* is False, steps 3 and 4 are skipped, and the
        mean is returned as a :class:`~orix.quaternion.Rotation` to
        signify the loss of symmetry information.

        Since a pure rotation cannot align an inverted reference frame
        with an uninverted one, a Frobenius norm cannot be calculated
        for a mix of proper and improper rotations. By default, this
        problem is addressed by ignoring symmetrically equivalent
        operations that include inversion. This aligns with the
        definition of a fundamental zone in (mis)orientation space used
        in :meth:`~orix.quaternion.Misorientation.reduce` and
        :meth:`~orix.quaternion.OrientationRegion.from_symmetry`.
        Passing True to *include_improper* will instead investigate all
        symmetry operations and treat all operations as proper when
        calculating the mean.
        """
        if not use_symmetry:
            # Convert to a rotation to emphasize loss of symmetry
            # information
            return Rotation(self.data).mean(weights=weights)

        start, end = self._symmetry
        if not include_improper:
            start = start.proper_subgroup
            end = end.proper_subgroup

        symmetry_pairs = iproduct(Rotation(start), Rotation(end))

        # Overwrite new nearest values into neighbors. Use rotations for
        # calculating the candidate for inclusion in neighbors.
        neighbors = self.reduce(verbose=verbose)
        rot = Rotation(neighbors.data)
        rough_mean = rot.mean(weights=weights)
        max_dp = rot.dot(rough_mean)

        log_msg = "Checking for closer equivalent representations"
        if verbose:
            symmetry_pairs = tqdm(
                symmetry_pairs, total=start.size * end.size, desc=log_msg
            )
        else:
            _logger.info(log_msg)

        for start, end in symmetry_pairs:
            candidates = end * rot * start
            dp = np.abs(candidates.dot(rough_mean))
            mask = dp > max_dp

            # Copy quaternion plus handedness
            neighbors._data[mask, :] = candidates._data[mask, :]

            max_dp[mask] = dp[mask]

        fine_mean = Rotation(neighbors.data).mean(weights=weights)
        fine_mean = self.__class__(fine_mean)
        fine_mean._symmetry = self._symmetry

        if return_neighbors:
            return fine_mean, neighbors

        return fine_mean


def _get_distance_matrix_dask(
    mori: Misorientation,
    symmetry: Symmetry,
    chunk_size: int,
    progressbar: bool,
) -> np.ndarray:
    # Perform "s_k m_i s_l s_k m_j" (see Notes)
    M1 = symmetry.outer(mori).outer(symmetry)
    M2 = M1._outer_dask(~mori, chunk_size=chunk_size)

    # Perform last outer product and reduce to all dot products at
    # the same time
    warnings.filterwarnings("ignore", category=da.PerformanceWarning)
    str1 = "abcdefghijklmnopqrstuvwxy"[: M2.ndim]
    str2 = "z" + str1[-1]  # Last axis has shape (4,)
    sum_over = f"{str1},{str2}->{str1[:-1] + str2[0]}"
    all_dot_products = da.einsum(sum_over, M2, symmetry.data)

    # Get highest dot product
    axes = (0, mori.ndim + 1, 2 * mori.ndim + 2)
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
