# -*- coding: utf-8 -*-
# Copyright 2018-2020 the orix developers
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

An orientation is simply a rotation with respect to some reference frame. In
this respect, an orientation is in fact a *misorientation* - a change of
orientation - with respect to a reference of the identity rotation.

In orix, orientations and misorientations are distinguished from rotations
only by the inclusion of a notion of symmetry. Consider the following example:

.. image:: /_static/img/orientation.png
   :width: 200px
   :alt: Two objects with two different rotations each. The square, with
         fourfold symmetry, has the same orientation in both cases.
   :align: center

Both objects have undergone the same *rotations* with respect to the reference.
However, because the square has fourfold symmetry, it is indistinguishable
in both cases, and hence has the same orientation.

"""

from itertools import product as iproduct
from itertools import combinations_with_replacement as icombinations
import numpy as np
import warnings
from tqdm import tqdm


from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C1
from orix.quaternion.orientation_region import OrientationRegion


def _distance(misorientation, verbose, split_size=100):
    """ private function to find the symmetry reduced distance between all
    pairs of (mis)orientations

    Parameters
    ----------
    misorientation : orix.Misorientation object
        The misorientation to be considered.
    verbose : bool
        Output progress bar while computing.
    split_size : int
        Size of block to compute at a time.

    Returns
    -------
    distance : np.array
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
    """Misorientation object.

    Misorientations represent transformations from one orientation,
    :math:`o_1` to another, :math:`o_2`: :math:`o_2 \\cdot o_1^{-1}`.

    They have symmetries associated with each of the starting orientations.

    """

    _symmetry = (C1, C1)

    def __getitem__(self, key):
        m = super(Misorientation, self).__getitem__(key)
        m._symmetry = self._symmetry
        return m

    @property
    def symmetry(self):
        """tuple of Symmetry"""
        return self._symmetry

    def equivalent(self, grain_exchange=False):
        """Equivalent misorientations

        grain_exchange : bool
            If true the rotation g and g^{-1} are considered to be identical

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

    def set_symmetry(self, Gl, Gr, verbose=False):
        """Assign symmetries to this misorientation.

        Computes equivalent transformations which have the smallest angle of
        rotation and assigns these in-place.

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
        """Symmetry reduced distance

        Compute the shortest distance between all orientations considering
        symmetries.

        Parameters
        ---------
        verbose : bool
            Output progress bar while computing.
        split_size : int
            Size of block to compute at a time.

        Returns
        -------
        distance : np.array
            2D matrix containing the angular distance between every
            orientation, considering symmetries.

        Examples
        --------
        >>> import numpy as np
        >>> from orix.quaternion.symmetry import C4, C2
        >>> from orix.quaternion.orientation import Misorientation
        >>> data = np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 0, 0]])
        >>> m = Misorientation(data).set_symmetry(C4, C2)
        >>> m.distance()
        array([[3.14159265, 1.57079633],
               [1.57079633, 0.        ]])
        """
        distance = _distance(self, verbose, split_size)
        return distance.reshape(self.shape + self.shape)

    def __repr__(self):
        cls = self.__class__.__name__
        shape = str(self.shape)
        s1, s2 = self._symmetry[0].name, self._symmetry[1].name
        s2 = "" if s2 == "1" else s2
        symm = s1 + (s2 and ", ") + s2
        data = np.array_str(self.data, precision=4, suppress_small=True)
        rep = "{} {} {}\n{}".format(cls, shape, symm, data)
        return rep


class Orientation(Misorientation):
    """Orientation object.

    Orientations represent misorientations away from a reference of identity
    and have only one associated symmetry.

    Orientations support binary subtraction, producing a misorientation. That
    is, to compute the misorientation from :math:`o_1` to :math:`o_2`,
    call :code:`o_2 - o_1`.

    """

    @property
    def symmetry(self):
        """Symmetry"""
        return self._symmetry[1]

    def set_symmetry(self, symmetry):
        """Assign a symmetry to this orientation.

        Computes equivalent transformations which have the smallest angle of
        rotation and assigns these in-place.

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
        >>> o = Orientation(data).set_symmetry((C4))
        >>> o
        Orientation (2,) 4
        [[-0.7071  0.     -0.7071  0.    ]
        [ 0.      1.      0.      0.    ]]

        """
        return super(Orientation, self).set_symmetry(C1, symmetry)

    def __sub__(self, other):
        if isinstance(other, Orientation):
            misorientation = Misorientation(self * ~other)
            m_inside = misorientation.set_symmetry(
                self.symmetry, other.symmetry
            ).squeeze()
            return m_inside
        return NotImplemented
