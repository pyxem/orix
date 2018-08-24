"""Rotations respecting symmetry.

An orientation is simply a rotation with respect to some reference frame. In
this respect, an orientation is in fact a *misorientation* - a change of
orientation - with respect to a reference of the identity rotation.

In texpy, orientations and misorientations are distinguished from rotations
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

import numpy as np

from texpy.quaternion.rotation import Rotation
from texpy.quaternion.symmetry import C1
from texpy.quaternion.orientation_region import OrientationRegion


class Misorientation(Rotation):
    """Misorientation object.

    Misorientations represent transformations from one orientation,
    :math:`o_1` to another, :math:`o_2`: :math:`o_2 \\cdot o_1^{-1}`.

    They have symmetries associated with each of the starting orientations.

    """

    _symmetry = (C1, C1)

    def __finalize__(self, data):
        super(Misorientation, self).__finalize__(data)
        if isinstance(data, Misorientation):
            self._symmetry = data._symmetry

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
        equivalent = Gr.outer(orientations.outer(Gl))
        return self.__class__(equivalent).flatten()

    def set_symmetry(self, symmetry):
        """Assign symmetries to this misorientation.

        Computes equivalent transformations which have the smallest angle of
        rotation and assigns these in-place.

        Parameters
        ----------
        symmetry : tuple of Symmetry

        Returns
        -------
        Misorientation
            The instance itself, with equivalent values.

        Examples
        --------
        >>> from texpy.quaternion.symmetry import C4, C2
        >>> data = np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 0, 0]])
        >>> m = Misorientation(data).set_symmetry((C4, C2))
        >>> m
        Misorientation (2,) 4, 2
        [[-0.7071  0.     -0.7071  0.    ]
         [ 0.      0.7071 -0.7071  0.    ]]

        """
        Gl, Gr = symmetry
        orientation_region = OrientationRegion.from_symmetry(*symmetry)
        o_inside = np.zeros_like(self.data)
        o_equivalent = Gr.outer(self.outer(Gl))
        inside = np.where(np.logical_and(o_equivalent < orientation_region, ~o_equivalent.improper))
        o_inside[inside[1:-1]] = o_equivalent[inside].data
        o_inside = self.__class__(o_inside)
        o_inside._symmetry = symmetry
        return o_inside

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
        return self._symmetry[0]

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
        >>> from texpy.quaternion.symmetry import C4
        >>> data = np.array([[0.5, 0.5, 0.5, 0.5], [0, 1, 0, 0]])
        >>> o = Orientation(data).set_symmetry((C4))
        >>> o
        Orientation (2,) 4
        [[-0.7071  0.     -0.7071  0.    ]
         [ 0.     -0.7071 -0.7071  0.    ]]

        """
        return super(Orientation, self).set_symmetry((symmetry, C1))

    def __sub__(self, other):
        if isinstance(other, Orientation):
            misorientation = Misorientation(self * ~other)
            m_inside = misorientation.set_symmetry((self.symmetry, other.symmetry)).squeeze()
            return m_inside
        return NotImplemented







