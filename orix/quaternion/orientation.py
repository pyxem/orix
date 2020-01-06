# -*- coding: utf-8 -*-
# Copyright 2018-2019 The pyXem developers
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
import numpy as np

from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C1
from orix.quaternion.orientation_region import OrientationRegion


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

    @property
    def equivalent(self):
        """Equivalent misorientations

        Returns
        -------
        Misorientation

        """
        Gl, Gr = self._symmetry
        if Gl._tuples == Gr._tuples:  # Grain exchange case
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
        [[-0.7071  0.     -0.7071  0.    ]
         [ 0.      0.7071 -0.7071  0.    ]]

        """
        symmetry_pairs = iproduct(Gl, Gr)
        if verbose:
            import tqdm
            symmetry_pairs = tqdm.tqdm(symmetry_pairs, total=Gl.size * Gr.size)
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

    def distance(self, speed=1, verbose=False):
        _distance_method = _distance_1
        if speed == 2:
            _distance_method = _distance_2
        distance = _distance_method(self, verbose)
        return distance.reshape(self.shape + self.shape)

    def __repr__(self):
        cls = self.__class__.__name__
        shape = str(self.shape)
        s1, s2 = self._symmetry[0].name, self._symmetry[1].name
        s2 = '' if s2 == '1' else s2
        symm = s1 + (s2 and ', ') + s2
        data = np.array_str(self.data, precision=4, suppress_small=True)
        rep = '{} {} {}\n{}'.format(cls, shape, symm, data)
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
         [ 0.     -0.7071 -0.7071  0.    ]]

        """
        return super(Orientation, self).set_symmetry(C1, symmetry)

    def __sub__(self, other):
        if isinstance(other, Orientation):
            misorientation = Misorientation(self * ~other)
            m_inside = misorientation.set_symmetry(self.symmetry, other.symmetry).squeeze()
            return m_inside
        return NotImplemented


def _distance_1(misorientation, verbose):
    from itertools import combinations_with_replacement as icombinations
    s_1, s_2 = misorientation._symmetry
    distance = np.empty((misorientation.size, misorientation.size))
    index_pairs = icombinations(range(misorientation.size), 2)
    if verbose:
        from tqdm import tqdm
        index_pairs = tqdm(index_pairs, total=misorientation.size ** 2)
    for i, j in index_pairs:
        idxi = np.unravel_index(i, misorientation.shape)
        idxj = np.unravel_index(j, misorientation.shape)
        m_1, m_2 = misorientation[idxi], misorientation[idxj]
        mis2orientation = (
            s_2.outer(~m_1).outer(s_1).outer(s_1).outer(m_2).outer(s_2)
        )

        axis = (0, len(misorientation.shape) + 1, len(misorientation.shape) + 2, -1)
        d = mis2orientation.angle.data.min(axis=axis)
        distance[i, j] = d
        distance[j, i] = d
    return distance


def _distance_2(misorientation, verbose):
    if misorientation.size > 1e4:  # pragma no cover
        confirm = input('Large datasets may crash your RAM.\nAre you sure? (y/n) ')
        if confirm != 'y':
            raise InterruptedError('Aborted')
    from itertools import product as iproduct
    S_1, S_2 = misorientation._symmetry
    mis2orientation = (~misorientation).outer(S_1).outer(S_1).outer(misorientation)
    distance = np.full(misorientation.shape + misorientation.shape, np.infty)
    symmetry_pairs = iproduct(S_2, S_2)
    if verbose:
        from tqdm import tqdm
        symmetry_pairs = tqdm(symmetry_pairs, total=S_2.size ** 2)
    for s_1, s_2 in symmetry_pairs:
        m = s_1 * mis2orientation * s_2
        axis = (len(misorientation.shape), len(misorientation.shape) + 1)
        angle = m.angle.data.min(axis=axis)
        distance = np.minimum(distance, angle)
    return distance
