import itertools

import tqdm
import numpy as np

from texpy.object3d import Object3d, check_matching_type
from texpy.vector.vector3d import Vector3d


class Quaternion(Object3d):

    dim = 4
    data = None

    def __neg__(self):
        return self.__class__(-self.data)

    @property
    def a(self):
        return self.data[..., 0]

    @a.setter
    def a(self, value):
        self.data[..., 0] = value

    @property
    def b(self):
        return self.data[..., 1]

    @b.setter
    def b(self, value):
        self.data[..., 1] = value

    @property
    def c(self):
        return self.data[..., 2]

    @c.setter
    def c(self, value):
        self.data[..., 2] = value

    @property
    def d(self):
        return self.data[..., 3]

    @d.setter
    def d(self, value):
        self.data[..., 3] = value

    @property
    def conj(self):
        a = self.a
        b, c, d = -self.b, -self.c, -self.d
        q = np.stack((a, b, c, d), axis=-1)
        return Quaternion(q)

    def __invert__(self):
        return self.__class__(self.conj.data / (self.norm ** 2)[..., np.newaxis])

    def outer(self, other):
        if isinstance(other, Quaternion):
            e = lambda x, y: np.multiply.outer(x, y)
            sa, oa = self.a, other.a
            sb, ob = self.b, other.b
            sc, oc = self.c, other.c
            sd, od = self.d, other.d
            a = e(sa, oa) - e(sb, ob) - e(sc, oc) - e(sd, od)
            b = e(sb, oa) + e(sa, ob) - e(sd, oc) + e(sc, od)
            c = e(sc, oa) + e(sd, ob) + e(sa, oc) - e(sb, od)
            d = e(sd, oa) - e(sc, ob) + e(sb, oc) + e(sa, od)
            q = np.stack((a, b, c, d), axis=-1)
            return other.__class__(q)
        return super(Quaternion, self).outer(other)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            sa, oa = self.a, other.a
            sb, ob = self.b, other.b
            sc, oc = self.c, other.c
            sd, od = self.d, other.d
            a = sa * oa - sb * ob - sc * oc - sd * od
            b = sb * oa + sa * ob - sd * oc + sc * od
            c = sc * oa + sd * ob + sa * oc - sb * od
            d = sd * oa - sc * ob + sb * oc + sa * od
            q = np.stack((a, b, c, d), axis=-1)
            return other.__class__(q)
        elif isinstance(other, Vector3d):
            a, b, c, d = self.a, self.b, self.c, self.d
            x, y, z = other.x, other.y, other.z
            x_new = (a ** 2 + b ** 2 - c ** 2 - d ** 2) * x + 2 * ((a * c + b * d) * z + (b * c - a * d) * y)
            y_new = (a ** 2 - b ** 2 + c ** 2 - d ** 2) * y + 2 * ((a * d + b * c) * x + (c * d - a * b) * z)
            z_new = (a ** 2 - b ** 2 - c ** 2 + d ** 2) * z + 2 * ((a * b + c * d) * y + (b * d - a * c) * x)
            return other.__class__(np.stack((x_new, y_new, z_new), axis=-1))
        return NotImplemented

    def dot(self, other):
        return np.sum(self.data * other.data, axis=-1)

    def dot_outer(self, other):
        dots = np.tensordot(self.data, other.data, axes=(-1, -1))
        return dots

    def angle_with(self, other):
        other = check_matching_type(self, other)
        angles = np.nan_to_num(np.arccos(2 * self.unit.dot(other.unit) ** 2 - 1))
        return angles

    @staticmethod
    def triple_cross(q1, q2, q3):
        a = + q1.b * q2.c * q3.d - q1.b * q3.c * q2.d - q2.b * q1.c * q3.d \
            + q2.b * q3.c * q1.d + q3.b * q1.c * q2.d - q3.b * q2.c * q1.d
        b = + q1.a * q3.c * q2.d - q1.a * q2.c * q3.d + q2.a * q1.c * q3.d \
            - q2.a * q3.c * q1.d - q3.a * q1.c * q2.d + q3.a * q2.c * q1.d
        c = + q1.a * q2.b * q3.d - q1.a * q3.b * q2.d - q2.a * q1.b * q3.d \
            + q2.a * q3.b * q1.d + q3.a * q1.b * q2.d - q3.a * q2.b * q1.d
        d = + q1.a * q3.b * q2.c - q1.a * q2.b * q3.c + q2.a * q1.b * q3.c \
            - q2.a * q3.b * q1.c - q3.a * q1.b * q2.c + q3.a * q2.b * q1.c
        q = Quaternion(np.vstack((a, b, c, d)).T)
        return q

    @property
    def axis(self):
        axis = Vector3d(np.stack((self.b, self.c, self.d), axis=-1))
        axis[self.a < -1e-6] = -axis[self.a < -1e-6]
        axis[axis.norm == 0] = Vector3d.xvector()
        axis.data = axis.data / axis.norm[..., np.newaxis]
        return axis

    @property
    def angle(self):
        return 2 * np.nan_to_num(np.arccos(np.abs(self.a)))

    def to_rotation(self):
        from .rotation import Rotation
        return Rotation(self.unit.data)

