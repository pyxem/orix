import numpy as np

from texpy.quaternion import Quaternion
from texpy.vector import Vector3d
from texpy.scalar import Scalar
from texpy.plot.rotation_plot import RotationPlot, plot_pole_figure


class Rotation(Quaternion):

    _improper = None
    plot_type = RotationPlot

    def __init__(self, data):
        super(Rotation, self).__init__(data)
        if isinstance(data, Rotation):
            self.improper = data.improper
        with np.errstate(divide='ignore', invalid='ignore'):
            self.data = self.data / self.norm.data[..., np.newaxis]

    def __mul__(self, other):
        if isinstance(other, Rotation):
            q = Quaternion(self) * Quaternion(other)
            r = Rotation(q)
            i = np.logical_xor(self.improper, other.improper)
            r.improper = i
            return r
        if isinstance(other, Quaternion):
            q = Quaternion(self) * other
            return q
        if isinstance(other, Vector3d):
            v = Quaternion(self) * other
            improper = (self.improper * np.ones(other.shape)).astype(bool)
            v[improper] = -v[improper]
            return v
        try:
            other = np.atleast_1d(other).astype(int)
        except ValueError:
            pass
        if isinstance(other, np.ndarray):
            assert np.all(abs(other) == 1), "Rotations can only be multiplied by 1 or -1"
            r = Rotation(self.data)
            r.improper = np.logical_xor(self.improper, other == -1)
            return r
        return NotImplemented

    def __neg__(self):
        r = self.__class__(self.data)
        r.improper = np.logical_not(self.improper)
        return r

    def __invert__(self):
        r = super(Rotation, self).__invert__()
        r.improper = self.improper
        return r

    def __getitem__(self, key):
        obj = super(Rotation, self).__getitem__(key)
        i = self.improper[key]
        obj.improper = np.atleast_1d(i)
        return obj

    def flatten(self):
        r = super(Rotation, self).flatten()
        r.improper = self.improper.flatten()
        return r

    def unique(self, return_index=False, return_inverse=False):
        if len(self.data) == 0:
            return self.__class__(self.data)
        rotation = self.flatten()
        a = rotation.a.data
        b = rotation.b.data
        c = rotation.c.data
        d = rotation.d.data
        i = rotation.improper
        abcd = np.stack((a ** 2, b ** 2, c ** 2, d ** 2, a * b, a * c, a * d,
                         b * c, b * d, c * d, i), axis=-1).round(5)
        _, idx, inv = np.unique(abcd, axis=0, return_index=True, return_inverse=True)
        dat = rotation[np.sort(idx)]
        if return_index and return_inverse:
            return dat, idx, inv
        elif return_index and not return_inverse:
            return dat, idx
        elif return_inverse and not return_index:
            return dat, inv
        else:
            return dat

    def angle_with(self, other):
        other = check_quaternion(other)
        angles = Scalar(np.nan_to_num(np.arccos(2 * self.unit.dot(other.unit).data ** 2 - 1)))
        return angles

    def outer(self, other):
        r = super(Rotation, self).outer(other)
        if isinstance(r, Rotation):
            r.improper = np.logical_xor.outer(self.improper, other.improper)
        if isinstance(r, Vector3d):
            r[self.improper] = -r[self.improper]
        return r

    def flatten(self):
        r = super(Rotation, self).flatten()
        r.improper = self.improper.T.flatten().T
        return r

    @property
    def improper(self):
        if self._improper is None:
            self._improper = np.zeros(self.shape, dtype=bool)
        return self._improper.astype(bool)

    @improper.setter
    def improper(self, value):
        value = np.atleast_1d(value)
        assert value.shape == self.shape, "Shape must be {}. (Gave {}).".format(self.shape, value.shape)
        self._improper = value

    def dot_outer(self, other):
        cosines = np.abs(super(Rotation, self).dot_outer(other).data)
        if isinstance(other, Rotation):
            improper = self.improper.reshape(self.shape + (1,) * len(other.shape))
            i = np.logical_xor(improper, other.improper)
            cosines = np.minimum(~i, cosines)
        else:
            cosines[self.improper] = 0
        return Scalar(cosines)

    @classmethod
    def from_neo_euler(cls, neo_euler):
        """Creates a rotation from a neo-euler (vector) representation.

        Parameters
        ----------
        neo_euler : NeoEuler

        Returns
        -------
        Rotation

        """
        s = np.sin(neo_euler.angle.data / 2)
        a = np.cos(neo_euler.angle.data / 2)
        b = s * neo_euler.axis.x.data
        c = s * neo_euler.axis.y.data
        d = s * neo_euler.axis.z.data
        r = cls(np.stack([a, b, c, d], axis=-1))
        return r

    def to_homochoric(self):
        angle = self.angle.data
        coefficient = ((angle - np.sin(angle)) * 0.75) ** (1/3)
        return self.axis * coefficient

    def to_euler(self, convention='bunge'):  # TODO: other conventions
        """Rotations as Euler angles.

        Parameters
        ----------
        convention : 'matthies' | 'bunge' | 'zxz'
            The Euler angle convention used.

        Returns
        -------
        ndarray
            Array of Euler angles in radians.

        """
        at1 = np.arctan2(self.d.data, self.a.data)
        at2 = np.arctan2(self.b.data, self.c.data)
        alpha = at1 - at2
        beta = 2 * np.arctan2(np.sqrt(self.b.data ** 2 + self.c.data ** 2),
                              np.sqrt(self.a.data ** 2 + self.d.data ** 2))
        gamma = at1 + at2
        mask = np.isclose(beta, 0)
        alpha[mask] = 2 * np.arcsin(
            np.maximum(-1, np.minimum(1, np.sign(self.a[mask].data) * self.d[mask].data)))
        gamma[mask] = 0

        if convention == 'bunge' or convention == 'zxz':
            mask = ~np.isclose(beta, 0)
            alpha[mask] += np.pi / 2
            gamma[mask] += 3 * np.pi / 2
        else:
            raise NotImplementedError(
                '{} is not an implemented convention. See docstring.'.format(
                    convention))

        alpha = np.mod(alpha, 2 * np.pi)
        gamma = np.mod(gamma, 2 * np.pi)

        return np.stack((alpha, beta, gamma), axis=-1)


    @classmethod
    def from_euler(cls, euler):
        # Bunge convention
        euler = np.array(euler)
        n = euler.shape[:-1]
        alpha, beta, gamma = euler[..., 0], euler[..., 1], euler[..., 2]
        alpha -= np.pi / 2
        gamma -= 3 * np.pi / 2
        zero = np.zeros(n)
        qalpha = Quaternion(
            np.stack((np.cos(alpha / 2), zero, zero, np.sin(alpha / 2)),
                     axis=-1))
        qbeta = Quaternion(
            np.stack((np.cos(beta / 2), zero, np.sin(beta / 2), zero), axis=-1))
        qgamma = Quaternion(
            np.stack((np.cos(gamma / 2), zero, zero, np.sin(gamma / 2)),
                     axis=-1))
        data = qalpha * qbeta * qgamma
        rot = Rotation(data.data)
        rot.improper = zero
        return rot

    @classmethod
    def identity(cls, N=1):
        return cls(np.hstack([np.ones((N, 1)), np.zeros((N, 3))]))

    @property
    def reciprocal(self):
        angles = np.zeros(self.shape) + np.pi
        return self * Rotation.from_axangle(-self.axis, angles)

    def min_axes(self):
        axes = self.axis
        angle = np.minimum(self.angle.data, 2 * np.pi - self.angle.data)
        print(angle)
        return axes[~np.isclose(angle, 0)]

    def plot_pole_figure(self, **kwargs):
        return plot_pole_figure(self, **kwargs)

    @property
    def axis(self):
        """Vector3d : the axis of rotation."""
        axis = Vector3d(np.stack((self.b.data, self.c.data, self.d.data), axis=-1))
        axis[self.a.data < -1e-6] = -axis[self.a.data < -1e-6]
        axis[axis.norm.data == 0] = Vector3d.xvector()
        axis.data = axis.data / axis.norm.data[..., np.newaxis]
        return axis

    @property
    def angle(self):
        """Scalar : the angle of rotation."""
        return Scalar(2 * np.nan_to_num(np.arccos(np.abs(self.a.data))))