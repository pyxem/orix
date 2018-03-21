import numpy as np
from math import sin, cos, pi

from texpy.point_group import PointGroup
from texpy.quaternion.orientation_region import OrientationRegion
from texpy.quaternion import Quaternion
from texpy.quaternion.rotation import Rotation
from texpy.vector.spherical_region import SphericalRegion
from texpy.vector import Vector3d


class Symmetry(Rotation):

    point_group = None
    fundamental_region = None

    def __init__(self, data):
        super(Symmetry, self).__init__(data)
        self.fundamental_region = self.calculate_fundamental_region()

    def __repr__(self):
        name = self.__class__.__name__
        point_group = str(self.point_group)
        data = np.array_str(self.data, precision=4, suppress_small=True)
        return '\n'.join([name + ' ' + point_group, data])

    def __getitem__(self, key):
        return self.to_rotation().__getitem__(key)

    @classmethod
    def from_symbol(cls, symmetry_symbol):
        point_group = PointGroup(symmetry_symbol)
        rotations = cls.rotations_from_point_group(point_group)
        s = cls(rotations.data)
        s.improper = rotations.improper
        s.point_group = point_group
        return s

    @staticmethod
    def rotations_from_point_group(point_group):
        if point_group.lattice in ['trigonal', 'hexagonal']:
            a, b, c = Vector3d((cos(- pi / 6), sin(- pi / 6), 0)), Vector3d.yvector(), Vector3d.zvector()
        else:
            a, b, c = Vector3d.xvector(), Vector3d.yvector(), Vector3d.zvector()
        rot = point_group.rotations(a, b, c)
        rotations = Rotation.identity()
        for r in rot:
            rotations = rotations.outer(r)
        return rotations.flatten()

    def calculate_fundamental_region(self):
        return OrientationRegion.from_symmetry(self)

    def calculate_fundamental_sector(self):
        return SphericalRegion.from_symmetry(self)

    @staticmethod
    def disjoint(symmetry_1, symmetry_2):
        is1, is2 = np.nonzero(np.isclose(symmetry_1.dot_outer(symmetry_2).data, 1))
        is1, is2 = np.array(tuple(set(is1))), np.array(tuple(set(is2)))
        if len(is1) == 1:
            return Symmetry.identity()
        if len(is1) == symmetry_1.size:
            return Symmetry(symmetry_1)
        if len(is2) == symmetry_2.size:
            return Symmetry(symmetry_2)
        s = Symmetry(symmetry_1[is1].data)
        s.improper = symmetry_1.improper[is1]
        return s

    def to_rotation(self):
        from .rotation import Rotation
        r = Rotation(self.data)
        r.improper = self.improper
        return r

    def symmetrise(self, quaternions):
        """Returns all rotations symmetrically equivalent to 'quaternions'.

        Parameters
        ----------
        quaternions : Quaternion

        Returns
        -------
        Quaternion

        """
        if isinstance(quaternions, Quaternion):
            q_related = self.to_rotation().outer(quaternions)
            return quaternions.__class__(q_related.data.reshape(q_related.shape[::-1] + (-1,)))
        if isinstance(quaternions, Vector3d):
            v_related = self.to_rotation().outer(quaternions)
            return quaternions.__class__(v_related.data.reshape(v_related.shape[::-1] + (-1,)))
        return NotImplemented

    @staticmethod
    def factorise(s1, s2):
        """Factorises 's1' and 's2' into `l`, `d`, `r`

        Parameters
        ----------
        s1, s2 : Symmetry

        Returns
        -------
        l, d, r : Quaternion
            `s1 == l.outer(d)`
            `s2 == d.outer(r)`

        """
        qs1, qs2 = s1.to_quaternion(), s2.to_quaternion()
        in1, in2 = np.where(np.isclose(np.abs(qs1.dot_outer(qs2).data), 1))
        d = qs1[in1]  # Common symmetries

        # Compute l
        l = Quaternion((1, 0, 0, 0))
        subs = np.isclose(np.abs((l.outer(d)).dot_outer(qs1).data), 1)
        c = np.any(subs, axis=tuple(range(len(subs.shape) - 1)))
        while not np.all(c):
            new = np.where(~c)[0][0]
            l.data = np.concatenate([l.data, qs1[new].data], axis=0)
            subs = np.isclose(np.abs(l.outer(d).dot_outer(qs1).data), 1)
            c = np.any(subs, axis=tuple(range(len(subs.shape) - 1)))

        # Compute r
        r = Quaternion((1, 0, 0, 0))
        c = np.any(np.isclose(np.abs((d * r).dot_outer(qs2).data), 1), axis=0)
        while not np.all(c):
            new = np.where(~c)[0][0]
            r.data = np.concatenate([r.data, qs2[new].data], axis=0)
            subs = np.isclose(np.abs(d.outer(r).dot_outer(qs2).data), 1)
            c = np.any(subs, axis=tuple(range(len(subs.shape) - 1)))

        return l, d, r




