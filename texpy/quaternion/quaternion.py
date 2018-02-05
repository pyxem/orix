import numpy as np

from texpy.base.object3d import Object3d, check
from texpy.scalar.scalar import Scalar
from texpy.vector.vector3d import Vector3d


def check_quaternion(obj):
    return check(obj, Quaternion)


class Quaternion(Object3d):

    dim = 4
    data = None

    @property
    def a(self):
        return Scalar(self.data[..., 0])

    @a.setter
    def a(self, value):
        self.data[..., 0] = value

    @property
    def b(self):
        return Scalar(self.data[..., 1])

    @b.setter
    def b(self, value):
        self.data[..., 1] = value

    @property
    def c(self):
        return Scalar(self.data[..., 2])

    @c.setter
    def c(self, value):
        self.data[..., 2] = value

    @property
    def d(self):
        return Scalar(self.data[..., 3])

    @d.setter
    def d(self, value):
        self.data[..., 3] = value

    @property
    def conj(self):
        a = self.a.data
        b, c, d = -self.b.data, -self.c.data, -self.d.data
        q = np.stack((a, b, c, d), axis=-1)
        return Quaternion(q)

    def __neg__(self):
        return self.__class__(-self.data)

    def __invert__(self):
        return self.__class__(self.conj.data / (self.norm.data ** 2)[..., np.newaxis])

    def outer(self, other):
        if isinstance(other, Quaternion):
            e = lambda x, y: np.multiply.outer(x, y)
            sa, oa = self.a.data, other.a.data
            sb, ob = self.b.data, other.b.data
            sc, oc = self.c.data, other.c.data
            sd, od = self.d.data, other.d.data
            a = e(sa, oa) - e(sb, ob) - e(sc, oc) - e(sd, od)
            b = e(sb, oa) + e(sa, ob) - e(sd, oc) + e(sc, od)
            c = e(sc, oa) + e(sd, ob) + e(sa, oc) - e(sb, od)
            d = e(sd, oa) - e(sc, ob) + e(sb, oc) + e(sa, od)
            q = np.stack((a, b, c, d), axis=-1)
            return other.__class__(q)
        return super(Quaternion, self).outer(other)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            sa, oa = self.a.data, other.a.data
            sb, ob = self.b.data, other.b.data
            sc, oc = self.c.data, other.c.data
            sd, od = self.d.data, other.d.data
            a = sa * oa - sb * ob - sc * oc - sd * od
            b = sb * oa + sa * ob - sd * oc + sc * od
            c = sc * oa + sd * ob + sa * oc - sb * od
            d = sd * oa - sc * ob + sb * oc + sa * od
            q = np.stack((a, b, c, d), axis=-1)
            return other.__class__(q)
        elif isinstance(other, Vector3d):
            a, b, c, d = self.a.data, self.b.data, self.c.data, self.d.data
            x, y, z = other.x.data, other.y.data, other.z.data
            x_new = (a ** 2 + b ** 2 - c ** 2 - d ** 2) * x + 2 * ((a * c + b * d) * z + (b * c - a * d) * y)
            y_new = (a ** 2 - b ** 2 + c ** 2 - d ** 2) * y + 2 * ((a * d + b * c) * x + (c * d - a * b) * z)
            z_new = (a ** 2 - b ** 2 - c ** 2 + d ** 2) * z + 2 * ((a * b + c * d) * y + (b * d - a * c) * x)
            return other.__class__(np.stack((x_new, y_new, z_new), axis=-1))
        return NotImplemented

    def dot(self, other):
        return Scalar(np.sum(self.data * other.data, axis=-1))

    def dot_outer(self, other):
        dots = np.tensordot(self.data, other.data, axes=(-1, -1))
        return Scalar(dots)

    def angle_with(self, other):
        other = check_quaternion(other)
        angles = Scalar(np.nan_to_num(np.arccos(2 * self.unit.dot(other.unit).data ** 2 - 1)))
        return angles

    @classmethod
    def triple_cross(cls, q1, q2, q3):
        q1a, q1b, q1c, q1d = q1.a.data, q1.b.data, q1.c.data, q1.d.data
        q2a, q2b, q2c, q2d = q2.a.data, q2.b.data, q2.c.data, q2.d.data
        q3a, q3b, q3c, q3d = q3.a.data, q3.b.data, q3.c.data, q3.d.data
        a = + q1b * q2c * q3d - q1b * q3c * q2d - q2b * q1c * q3d \
            + q2b * q3c * q1d + q3b * q1c * q2d - q3b * q2c * q1d
        b = + q1a * q3c * q2d - q1a * q2c * q3d + q2a * q1c * q3d \
            - q2a * q3c * q1d - q3a * q1c * q2d + q3a * q2c * q1d
        c = + q1a * q2b * q3d - q1a * q3b * q2d - q2a * q1b * q3d \
            + q2a * q3b * q1d + q3a * q1b * q2d - q3a * q2b * q1d
        d = + q1a * q3b * q2c - q1a * q2b * q3c + q2a * q1b * q3c \
            - q2a * q3b * q1c - q3a * q1b * q2c + q3a * q2b * q1c
        q = cls(np.vstack((a, b, c, d)).T)
        return q

    @property
    def axis(self):
        axis = Vector3d(np.stack((self.b.data, self.c.data, self.d.data), axis=-1))
        axis[self.a.data < -1e-6] = -axis[self.a.data < -1e-6]
        axis[axis.norm.data == 0] = Vector3d.xvector()
        axis.data = axis.data / axis.norm.data[..., np.newaxis]
        return axis

    @property
    def angle(self):
        return Scalar(2 * np.nan_to_num(np.arccos(np.abs(self.a.data))))

    def to_rotation(self):
        from .rotation import Rotation
        return Rotation(self.unit.data)

