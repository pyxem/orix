import numpy as np

from texpy.quaternion.rotation import Rotation
from texpy.quaternion.symmetry import C1
from texpy.quaternion.orientation_region import OrientationRegion


class Misorientation(Rotation):

    _symmetry = (C1, C1)

    def __finalize__(self, data):
        super(Misorientation, self).__finalize__(data)
        if isinstance(data, Misorientation):
            self._symmetry = data._symmetry

    @property
    def symmetry(self):
        return self._symmetry

    def set_symmetry(self, symmetry):
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

    @property
    def symmetry(self):
        return self._symmetry[0]

    def set_symmetry(self, symmetry):
        return super(Orientation, self).set_symmetry((symmetry, C1))

    def __sub__(self, other):
        if isinstance(other, Orientation):
            misorientation = Misorientation(~self * other)
            m_inside = misorientation.set_symmetry((self.symmetry, other.symmetry)).squeeze()
            return m_inside
        return NotImplemented







