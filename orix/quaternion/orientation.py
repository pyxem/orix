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

"""Rotations respecting symmetry.

An orientation is simply a rotation with respect to some reference
frame. In this respect, an orientation is in fact a *misorientation* -
a change of orientation - with respect to a reference of the identity
rotation.

In orix, orientations and misorientations are distinguished from
rotations only by the inclusion of a notion of symmetry. Consider the
following example:

.. image:: /_static/img/orientation.png
   :width: 200px
   :alt: Two objects with two different rotations each. The square, with
         fourfold symmetry, has the same orientation in both cases.
   :align: center

Both objects have undergone the same *rotations* with respect to the
reference. However, because the square has fourfold symmetry, it is
indistinguishable in both cases, and hence has the same orientation.
"""

from itertools import product as iproduct
from itertools import combinations_with_replacement as icombinations
import warnings

import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
from tqdm import tqdm

from orix.quaternion.orientation_region import OrientationRegion
from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C1, Symmetry
from orix.scalar import Scalar
from orix.vector import AxAngle
from orix._util import deprecated


def _distance(misorientation, verbose, split_size=100):
    """Private function to find the symmetry reduced distance between
    all pairs of (mis)orientations

    Parameters
    ----------
    misorientation : orix.quaternion.Misorientation
        The misorientation to be considered.
    verbose : bool
        Output progress bar while computing.
    split_size : int
        Size of block to compute at a time.

    Returns
    -------
    distance : numpy.ndarray
        2D matrix containing the angular distance between every
        orientation, considering symmetries.
    """
    num_orientations = misorientation.shape[0]
    S_1, S_2 = misorientation._symmetry
    distance = np.full(misorientation.shape + misorientation.shape, np.infty)
    split_size = split_size // S_1.shape[0]
    outer_range = range(0, num_orientations, split_size)
    if verbose:
        outer_range = tqdm(outer_range, total=np.ceil(num_orientations / split_size))

    S_1_outer_S_1 = S_1.outer(S_1)

    # Calculate the upper half of the distance matrix block by block
    for start_index_b in outer_range:
        # we use slice object for compactness
        index_slice_b = slice(
            start_index_b, min(num_orientations, start_index_b + split_size)
        )
        o_sub_b = misorientation[index_slice_b]
        for start_index_a in range(0, start_index_b + split_size, split_size):
            index_slice_a = slice(
                start_index_a, min(num_orientations, start_index_a + split_size)
            )
            o_sub_a = misorientation[index_slice_a]
            axis = (len(o_sub_a.shape), len(o_sub_a.shape) + 1)
            mis2orientation = (~o_sub_a).outer(S_1_outer_S_1).outer(o_sub_b)
            # This works through all the identity rotations
            for s_2_1, s_2_2 in icombinations(S_2, 2):
                m = s_2_1 * mis2orientation * s_2_2
                angle = m.angle.data.min(axis=axis)
                distance[index_slice_a, index_slice_b] = np.minimum(
                    distance[index_slice_a, index_slice_b], angle
                )
    # Symmetrize the matrix for convenience
    i_lower = np.tril_indices(distance.shape[0], -1)
    distance[i_lower] = distance.T[i_lower]
    return distance


class Misorientation(Rotation):
    r"""Misorientation object.

    Misorientations represent transformations from one orientation,
    :math:`o_1` to another, :math:`o_2`: :math:`o_2 \cdot o_1^{-1}`.

    They have symmetries associated with each of the starting
    orientations.
    """

    _symmetry = (C1, C1)

    def __init__(self, data, symmetry=None):
        super().__init__(data)
        if symmetry:
            self.symmetry = symmetry

    @property
    def symmetry(self):
        """Tuple of :class:`~orix.quaternion.Symmetry`."""
        return self._symmetry

    @symmetry.setter
    def symmetry(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError("Value must be a 2-tuple of Symmetry objects.")
        if len(value) != 2 or not all(isinstance(s, Symmetry) for s in value):
            raise ValueError("Value must be a 2-tuple of Symmetry objects.")
        self._symmetry = tuple(value)

    def __getitem__(self, key):
        m = super().__getitem__(key)
        m._symmetry = self._symmetry
        return m

    def reshape(self, *shape):
        m = super().reshape(*shape)
        m._symmetry = self._symmetry
        return m

    def flatten(self):
        m = super().flatten()
        m._symmetry = self._symmetry
        return m

    def squeeze(self):
        m = super().squeeze()
        m._symmetry = self._symmetry
        return m

    def transpose(self, *axes):
        m = super().transpose(*axes)
        m._symmetry = self._symmetry
        return m

    def equivalent(self, grain_exchange=False):
        r"""Equivalent misorientations.

        grain_exchange : bool
            If True the rotation $g$ and $g^{-1}$ are considered to be
            identical. Default is False.

        Returns
        -------
        Misorientation
        """
        Gl, Gr = self._symmetry

        if grain_exchange and (Gl._tuples == Gr._tuples):
            orientations = Orientation.stack([self, ~self]).flatten()
        else:
            orientations = Orientation(self)

        equivalent = Gr.outer(orientations.outer(Gl))
        return self.__class__(equivalent).flatten()

    @deprecated(
        since="0.8",
        alternative="orix.quaternion.Misorientation.map_into_symmetry_reduced_zone",
        removal="0.9",
    )
    def set_symmetry(self, Gl, Gr, verbose=False):
        """Assign symmetries to this misorientation.

        Computes equivalent transformations which have the smallest
        angle of rotation and assigns these in-place.

        Parameters
        ----------
        Gl, Gr : Symmetry

        Returns
        -------
        Misorientation
            A new misorientation object with the assigned symmetry.

        Examples
        --------
        >>> from orix.quaternion.symmetry import C4, C2
        >>> data = np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 0, 0]])
        >>> m = Misorientation(data).set_symmetry(C4, C2)
        >>> m
        Misorientation (2,) 4, 2
        [[-0.7071  0.7071  0.      0.    ]
        [ 0.      1.      0.      0.    ]]
        """
        misori = self.__class__(self.data)
        misori.symmetry = (Gl, Gr)
        return misori.map_into_symmetry_reduced_zone()

    def map_into_symmetry_reduced_zone(self, verbose=False):
        """Computes equivalent transformations which have the smallest
        angle of rotation and return these as a new Misorientation object.

        Returns
        -------
        Misorientation
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

    def distance(self, verbose=False, split_size=100):
        """Symmetry reduced distance.

        Compute the shortest distance between all orientations
        considering symmetries.

        Parameters
        ---------
        verbose : bool
            Output progress bar while computing. Default is False.
        split_size : int
            Size of block to compute at a time. Default is 100.

        Returns
        -------
        distance : numpy.ndarray
            2D matrix containing the angular distance between every
            orientation, considering symmetries.

        Examples
        --------
        >>> import numpy as np
        >>> from orix.quaternion import Misorientation, symmetry
        >>> data = np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 0, 0]])
        >>> m = Misorientation(data)
        >>> m.symmetry = (symmetry.C4, symmetry.C2)
        >>> m = m.map_into_symmetry_reduced_zone()
        >>> m.distance()
        array([[3.14159265, 1.57079633],
               [1.57079633, 0.        ]])
        """
        distance = _distance(self, verbose, split_size)
        return distance.reshape(self.shape + self.shape)

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
        projection="axangle",
        figure=None,
        position=None,
        return_figure=False,
        wireframe_kwargs=None,
        size=None,
        **kwargs,
    ):
        """Plot misorientations in axis-angle space or the Rodrigues
        fundamental zone.

        Parameters
        ----------
        projection : str, optional
            Which misorientation space to plot misorientations in,
            either "axangle" (default) or "rodrigues".
        figure : matplotlib.figure.Figure
            If given, a new plot axis :class:`~orix.plot.AxAnglePlot` or
            :class:`~orix.plot.RodriguesPlot` is added to the figure in
            the position specified by `position`. If not given, a new
            figure is created.
        position : int, tuple of int, matplotlib.gridspec.SubplotSpec,
                optional
            Where to add the new plot axis. 121 or (1, 2, 1) places it
            in the first of two positions in a grid of 1 row and 2
            columns. See :meth:`~matplotlib.figure.Figure.add_subplot`
            for further details. Default is (1, 1, 1).
        return_figure : bool, optional
            Whether to return the figure. Default is False.
        wireframe_kwargs : dict, optional
            Keyword arguments passed to
            :meth:`orix.plot.AxAnglePlot.plot_wireframe` or
            :meth:`orix.plot.RodriguesPlot.plot_wireframe`.
        size : int, optional
            If not given, all misorientations are plotted. If given, a
            random sample of this `size` of the misorientations is
            plotted.
        kwargs
            Keyword arguments passed to
            :meth:`orix.plot.AxAnglePlot.scatter` or
            :meth:`orix.plot.RodriguesPlot.scatter`.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure with the added plot axis, if `return_figure` is True.

        See Also
        --------
        orix.plot.AxAnglePlot, orix.plot.RodriguesPlot
        """
        from orix.plot.rotation_plot import _setup_rotation_plot

        figure, ax = _setup_rotation_plot(
            figure=figure, projection=projection, position=position
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


class Orientation(Misorientation):
    """Orientations represent misorientations away from a reference of
    identity and have only one associated symmetry.

    Orientations support binary subtraction, producing a misorientation.
    That is, to compute the misorientation from :math:`o_1` to
    :math:`o_2`, call :code:`o_2 - o_1`.
    """

    @property
    def symmetry(self):
        """Symmetry."""
        return self._symmetry[1]

    @symmetry.setter
    def symmetry(self, value):
        if not isinstance(value, Symmetry):
            raise TypeError("Value must be an instance of orix.quaternion.Symmetry.")
        self._symmetry = (C1, value)

    @property
    def unit(self):
        """Unit orientations."""
        o = super().unit
        o.symmetry = self.symmetry
        return o

    def __invert__(self):
        o = super().__invert__()
        o.symmetry = self.symmetry
        return o

    def __neg__(self):
        o = super().__neg__()
        o.symmetry = self.symmetry
        return o

    def __repr__(self):
        """String representation."""
        data = np.array_str(self.data, precision=4, suppress_small=True)
        return f"{self.__class__.__name__} {self.shape} {self.symmetry.name}\n{data}"

    def __sub__(self, other):
        if isinstance(other, Orientation):
            # Call to Object3d.squeeze() doesn't carry over symmetry
            misorientation = Misorientation(self * ~other).squeeze()
            misorientation.symmetry = (self.symmetry, other.symmetry)
            return misorientation.map_into_symmetry_reduced_zone()
        return NotImplemented

    @classmethod
    def from_euler(
        cls, euler, symmetry=None, convention="bunge", direction="crystal2lab"
    ):
        """Creates orientation(s) from an array of Euler angles.

        Parameters
        ----------
        euler : array-like
            Euler angles in the Bunge convention.
        symmetry : Symmetry, optional
            Symmetry of orientation(s). If None (default), no symmetry
            is set.
        convention : str
            Only 'bunge' is currently supported for new data
        direction : str
            'lab2crystal' or 'crystal2lab'
        """
        o = super().from_euler(euler=euler, convention=convention, direction=direction)
        if symmetry:
            o.symmetry = symmetry
        return o

    @classmethod
    def from_matrix(cls, matrix, symmetry=None):
        """Creates orientation(s) from orientation matrices
        :cite:`rowenhorst2015consistent`.

        Parameters
        ----------
        matrix : array_like
            Array of orientation matrices.
        symmetry : Symmetry, optional
            Symmetry of orientation(s). If None (default), no symmetry
            is set.
        """
        o = super().from_matrix(matrix)
        if symmetry:
            o.symmetry = symmetry
        return o

    @classmethod
    def from_neo_euler(cls, neo_euler, symmetry=None):
        """Creates orientation(s) from a neo-euler (vector)
        representation.

        Parameters
        ----------
        neo_euler : NeoEuler
            Vector parametrization of orientation(s).
        symmetry : Symmetry, optional
            Symmetry of orientation(s). If None (default), no symmetry
            is set.
        """
        o = super().from_neo_euler(neo_euler)
        if symmetry:
            o.symmetry = symmetry
        return o

    @classmethod
    def from_axes_angles(cls, axes, angles, symmetry=None):
        """Creates orientation(s) from axis-angle pair(s).

        Parameters
        ----------
        axes : Vector3d or array_like
            The axis of rotation.
        angles : array_like
            The angle of rotation, in radians.
        symmetry : Symmetry, optional
            Symmetry of orientation(s). If None (default), no symmetry
            is set.

        Returns
        -------
        Orientation

        Examples
        --------
        >>> import numpy as np
        >>> from orix.quaternion import Orientation, symmetry
        >>> ori = Orientation.from_axes_angles((0, 0, -1), np.pi / 2, symmetry.Oh)
        >>> ori
        Orientation (1,) m-3m
        [[ 0.7071  0.      0.     -0.7071]]

        See Also
        --------
        from_neo_euler
        """
        axangle = AxAngle.from_axes_angles(axes, angles)
        return cls.from_neo_euler(axangle, symmetry)

    def angle_with(self, other):
        """The smallest symmetry reduced angle of rotation transforming
        this orientation to the other.

        Parameters
        ----------
        other : orix.quaternion.Orientation

        Returns
        -------
        Scalar
        """
        dot_products = self.unit.dot(other.unit).data
        angles = np.nan_to_num(np.arccos(2 * dot_products ** 2 - 1))
        return Scalar(angles)

    def dot(self, other):
        """Symmetry reduced dot product of orientations in this instance
        to orientations in another instance, returned as
        :class:`~orix.scalar.Scalar`.

        See Also
        --------
        dot_outer
        """
        symmetry = self.symmetry.outer(other.symmetry).unique()
        misorientation = (~self) * other
        all_dot_products = Rotation(misorientation).dot_outer(symmetry).data
        highest_dot_product = np.max(all_dot_products, axis=-1)
        return Scalar(highest_dot_product)

    def dot_outer(self, other):
        """Symmetry reduced dot product of every orientation in this
        instance to every orientation in another instance, returned as
        :class:`~orix.scalar.Scalar`.

        See Also
        --------
        dot
        """
        symmetry = self.symmetry.outer(other.symmetry).unique()
        misorientation = (~self).outer(other)
        all_dot_products = Rotation(misorientation).dot_outer(symmetry).data
        highest_dot_product = np.max(all_dot_products, axis=-1)
        return Scalar(highest_dot_product)

    @deprecated(
        since="0.7",
        alternative="orix.quaternion.Orientation.get_distance_matrix",
        removal="0.8",
    )
    def distance(self, verbose=False, split_size=100):
        return super().distance(verbose=verbose, split_size=split_size)

    def get_distance_matrix(self, lazy=False, chunk_size=20, progressbar=True):
        r"""The symmetry reduced smallest angle of rotation transforming
        each orientation in this instance to every other orientation.

        This is an alternative implementation of
        :meth:`~orix.quaternion.Misorientation.distance` for
        a single :class:`Orientation` instance, using :mod:`dask`.

        Parameters
        ----------
        lazy : bool, optional
            Whether to perform the computation lazily with Dask. Default
            is False.
        chunk_size : int, optional
            Number of orientations per axis to include in each iteration
            of the computation. Default is 20. Only applies when `lazy`
            is True.
        progressbar : bool, optional
            Whether to show a progressbar during computation if `lazy`
            is True. Default is True.

        Returns
        -------
        Scalar

        Notes
        -----
        Given two orientations :math:`g_i` and :math:`g_j`, the smallest
        angle is considered as the geodesic distance

        .. math::

            d(g_i, g_j) = \arccos(2(g_i \cdot g_j)^2 - 1),

        where :math:`(g_i \cdot g_j)` is the highest dot product between
        symmetrically equivalent orientations to :math:`g_{i,j}`.
        """
        ori = self.unit
        if lazy:
            dot_products = ori._dot_outer_dask(ori, chunk_size=chunk_size)

            # Round because some dot products are slightly above 1
            n_decimals = np.finfo(dot_products.dtype).precision
            dot_products = da.round(dot_products, n_decimals)

            angles_dask = da.arccos(2 * dot_products ** 2 - 1)
            angles_dask = da.nan_to_num(angles_dask)

            # Create array in memory and overwrite, chunk by chunk
            angles = np.zeros(angles_dask.shape)
            if progressbar:
                with ProgressBar():
                    da.store(sources=angles_dask, targets=angles)
            else:
                da.store(sources=angles_dask, targets=angles)
        else:
            dot_products = ori.dot_outer(ori).data
            angles = np.arccos(2 * dot_products ** 2 - 1)
            angles = np.nan_to_num(angles)

        return Scalar(angles)

    def plot_unit_cell(
        self,
        c="tab:blue",
        return_figure=False,
        axes_length=0.5,
        structure=None,
        crystal_axes_loc="origin",
        **arrow_kwargs,
    ):
        """Plot the unit cell orientation, showing the sample and
        crystal reference frames.

        Parameters
        ----------
        c : str, optional
            Unit cell edge color.
        return_figure : bool, optional
            Return the plotted figure.
        axes_length : float, optional
            Length of the reference axes in Angstroms, by default 0.5.
        structure : diffpy.structure.Structure or None, optional
            Structure of the unit cell, only orthorhombic lattices are currently
            supported. If not given, a cubic unit cell with a lattice parameter of
            2 Angstroms will be plotted.
        crystal_axes_loc : str, optional
            Plot the crystal reference frame axes at the "origin" (default) or
            "center" of the plotted cell.
        arrow_kwargs : dict, optional
            Keyword arguments passed to
            :class:`matplotlib.patches.FancyArrowPatch`, for example "arrowstyle".

        Returns
        -------
        fig: matplotlib.figure.Figure
            The plotted figure.

        Raises
        ------
        ValueError
            If self.size > 1.
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

    @deprecated(
        since="0.8",
        alternative="orix.quaternion.Orientation.map_into_symmetry_reduced_zone",
        removal="0.9",
    )
    def set_symmetry(self, symmetry):
        """Assign a symmetry to this orientation.

        Computes equivalent transformations which have the smallest
        angle of rotation and assigns these in-place.

        Parameters
        ----------
        symmetry : Symmetry

        Returns
        -------
        Orientation
            The instance itself, with equivalent values.

        Examples
        --------
        >>> from orix.quaternion.symmetry import C4
        >>> data = np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 0, 0]])
        >>> o = Orientation(data).set_symmetry(C4)
        >>> o
        Orientation (2,) 4
        [[-0.7071  0.     -0.7071  0.    ]
        [ 0.      1.      0.      0.    ]]
        """
        o = self.__class__(self.data)
        o.symmetry = symmetry
        return o.map_into_symmetry_reduced_zone()

    def in_euler_fundamental_region(self):
        """Euler angles in the fundamental Euler region of the proper
        subgroup.

        The Euler angle ranges of each proper subgroup are given in
        :attr:`~orix.quaternion.Symmetry.euler_fundamental_region`.

        From the procedure in MTEX' :code:`quaternion.project2EulerFR`.

        Returns
        -------
        euler_in_region : numpy.ndarray
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
        projection="axangle",
        figure=None,
        position=None,
        return_figure=False,
        wireframe_kwargs=None,
        size=None,
        direction=None,
        **kwargs,
    ):
        """Plot orientations in axis-angle space, the Rodrigues
        fundamental zone, or an inverse pole figure (IPF) given a sample
        direction.

        Parameters
        ----------
        projection : str, optional
            Which orientation space to plot orientations in, either
            "axangle" (default), "rodrigues" or "ipf" (inverse pole
            figure).
        figure : matplotlib.figure.Figure
            If given, a new plot axis :class:`~orix.plot.AxAnglePlot` or
            :class:`~orix.plot.RodriguesPlot` is added to the figure in
            the position specified by `position`. If not given, a new
            figure is created.
        position : int, tuple of int, matplotlib.gridspec.SubplotSpec,
                optional
            Where to add the new plot axis. 121 or (1, 2, 1) places it
            in the first of two positions in a grid of 1 row and 2
            columns. See :meth:`~matplotlib.figure.Figure.add_subplot`
            for further details. Default is (1, 1, 1).
        return_figure : bool, optional
            Whether to return the figure. Default is False.
        wireframe_kwargs : dict, optional
            Keyword arguments passed to
            :meth:`orix.plot.AxAnglePlot.plot_wireframe` or
            :meth:`orix.plot.RodriguesPlot.plot_wireframe`.
        size : int, optional
            If not given, all orientations are plotted. If given, a
            random sample of this `size` of the orientations is plotted.
        direction : Vector3d, optional
            Sample direction to plot with respect to crystal directions.
            If not given, the out of plane direction, sample Z, is used.
            Only used when plotting IPF(s).
        kwargs
            Keyword arguments passed to
            :meth:`orix.plot.AxAnglePlot.scatter`,
            :meth:`orix.plot.RodriguesPlot.scatter`, or
            :meth:`orix.plot.InversePoleFigurePlot.scatter`.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure with the added plot axis, if `return_figure` is True.

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
                if np.any(sector.vertices.polar.data > np.pi / 2):
                    hemisphere = "both"
                else:
                    hemisphere = "upper"

                figure, axes = _setup_inverse_pole_figure_plot(
                    symmetry=symmetry, direction=direction, hemisphere=hemisphere
                )
            else:
                axes = np.asarray(figure.axes)

            for ax in axes:
                ax.scatter(self, **kwargs)

            figure.tight_layout()

        if return_figure:
            return figure

    def _dot_outer_dask(self, other, chunk_size=20):
        """Symmetry reduced dot product of every orientation in this
        instance to every orientation in another instance, returned as a
        Dask array.

        Parameters
        ----------
        other : orix.quaternion.Orientation
        chunk_size : int, optional
            Number of orientations per axis in each orientation instance
            to include in each iteration of the computation. Default is
            20.

        Returns
        -------
        dask.array.Array

        Notes
        -----
        To read the dot products array `dparr` into memory, do
        `dp = dparr.compute()`.
        """
        symmetry = self.symmetry.outer(other.symmetry).unique()
        misorientation = (~self)._outer_dask(other, chunk_size=chunk_size)

        # Summation subscripts
        str1 = "abcdefghijklmnopqrstuvwxy"[: misorientation.ndim]
        str2 = "z" + str1[-1]  # Last elements have shape (4,)
        sum_over = f"{str1},{str2}->{str1[:-1] + str2[0]}"

        warnings.filterwarnings("ignore", category=da.PerformanceWarning)

        all_dot_products = da.einsum(sum_over, misorientation, symmetry.data)
        highest_dot_product = da.max(abs(all_dot_products), axis=-1)

        return highest_dot_product
