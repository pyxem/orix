"""Point transformations of objects.

Rotations are transformations of three-dimensional space leaving the
origin in place. Rotations can be parametrized numerous ways, but in texpy are
handled as unit quaternions. Rotations can act on vectors, or other rotations,
but not scalars. They are often most easily visualised as being a turn of a
certain angle about a certain axis.

.. image:: /_static/img/rotation.png
   :width: 200px
   :alt: Rotation of an object illustrated with an axis and rotation angle.
   :align: center

Rotations can also be *improper*. An improper rotation in texpy operates on
vectors as a rotation by the unit quaternion, followed by inversion. Hence,
a mirroring through the x-y plane can be considered an improper rotation of
180° about the z-axis, illustrated in the figure below.

.. image:: /_static/img/inversion.png
   :width: 200px
   :alt: 180° rotation followed by inversion, leading to a mirror operation.
   :align: center


"""
import numpy as np
from scipy.special import hyp0f1

from texpy.quaternion import Quaternion
from texpy.vector import Vector3d
from texpy.scalar import Scalar


class Rotation(Quaternion):
    """Rotation object.

    Rotations support the following mathematical operations:

    - Unary negation.
    - Inversion.
    - Multiplication with other rotations and vectors.

    Rotations inherit all methods from :class:`Quaternion` although behaviour is
    different in some cases.

    Rotations can be converted to other parametrizations, notably the neo-Euler
    representations. See :class:`NeoEuler`.

    """

    def __init__(self, data):
        super(Rotation, self).__init__(data)
        self._data = np.concatenate((self.data, np.zeros(self.shape + (1,))), axis=-1)
        if isinstance(data, Rotation):
            self.improper = data.improper
        with np.errstate(divide='ignore', invalid='ignore'):
            self.data = self.data / self.norm.data[..., np.newaxis]

    def __mul__(self, other):
        if isinstance(other, Rotation):
            q = Quaternion(self) * Quaternion(other)
            r = other.__class__(q)
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

    def __getitem__(self, key):
        r = super(Rotation, self).__getitem__(key)
        r.improper = self.improper[key]
        return r

    def __invert__(self):
        r = super(Rotation, self).__invert__()
        r.improper = self.improper
        return r

    def unique(self, return_index=False, return_inverse=False, antipodal=True):
        """Returns a new object containing only this object's unique entries.

        Two rotations are not unique if:

            - they have the same propriety AND
                - they have the same numerical value OR
                - the numerical value of one is the negative of the other

        Parameters
        ----------
        return_index : bool, optional
            If True, will also return the indices of the (flattened) data where
            the unique entries were found.
        return_inverse : bool, optional
            If True, will also return the indices to reconstruct the (flattened)
            data from the unique data.
        antipodal : bool, optional
            If False, rotations representing the same transformation
            whose values are numerically different (negative) will *not* be
            considered unique.

        """
        if len(self.data) == 0:
            return self.__class__(self.data)
        rotation = self.flatten()
        if antipodal:
            abcd = rotation._differentiators()
        else:
            abcd = np.stack([rotation.a.data, rotation.b.data, rotation.c.data, rotation.d.data, rotation.improper], axis=-1).round(6)
        _, idx, inv = np.unique(abcd, axis=0, return_index=True, return_inverse=True)
        idx_sort = np.sort(idx)
        dat = rotation[idx_sort]
        dat.improper = rotation.improper[idx_sort]
        if return_index and return_inverse:
            return dat, idx_sort, inv
        elif return_index and not return_inverse:
            return dat, idx_sort
        elif return_inverse and not return_index:
            return dat, inv
        else:
            return dat

    def _differentiators(self):
        a = self.a.data
        b = self.b.data
        c = self.c.data
        d = self.d.data
        i = self.improper
        abcd = np.stack((a ** 2, b ** 2, c ** 2, d ** 2, a * b, a * c, a * d,
                         b * c, b * d, c * d, i), axis=-1).round(6)
        return abcd

    def angle_with(self, other):
        """The angle of rotation transforming this rotation to the other.

        Returns
        -------
        Scalar

        """
        other = Rotation(other)
        angles = Scalar(np.nan_to_num(np.arccos(2 * self.unit.dot(other.unit).data ** 2 - 1)))
        return angles

    def outer(self, other):
        """Compute the outer product of this rotation and the other object."""
        r = super(Rotation, self).outer(other)
        if isinstance(r, Rotation):
            r.improper = np.logical_xor.outer(self.improper, other.improper)
        if isinstance(r, Vector3d):
            r[self.improper] = -r[self.improper]
        return r

    def flatten(self):
        """A new object with the same data in a single column."""
        r = super(Rotation, self).flatten()
        r.improper = self.improper.T.flatten().T
        return r

    @property
    def improper(self):
        """ndarray : True for improper rotations and False otherwise."""
        return self._data[..., -1].astype(bool)

    @improper.setter
    def improper(self, value):
        self._data[..., -1] = value

    def dot_outer(self, other):
        """Scalar : the outer dot product of this rotation and the other."""
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
            Vector parametrization of a rotation.

        """
        s = np.sin(neo_euler.angle.data / 2)
        a = np.cos(neo_euler.angle.data / 2)
        b = s * neo_euler.axis.x.data
        c = s * neo_euler.axis.y.data
        d = s * neo_euler.axis.z.data
        r = cls(np.stack([a, b, c, d], axis=-1))
        return r

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
            np.maximum(-1, np.minimum(1, np.sign(self.a.data[mask]) * self.d.data[mask])))
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
        """Creates a rotation from an array of Euler angles.

        Parameters
        ----------
        euler : array-like
            Euler angles in the Bunge convention.

        """
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
        rot = cls(data.data)
        rot.improper = zero
        return rot

    @classmethod
    def identity(cls, shape=(1,)):
        """Create identity rotations.

        Parameters
        ----------
        shape : tuple
            The shape out of which to construct identity quaternions

        """
        data = np.zeros(shape + (4,))
        data[..., 0] = 1
        return cls(data)

    @property
    def axis(self):
        """Vector3d : the axis of rotation."""
        axis = Vector3d(np.stack((self.b.data, self.c.data, self.d.data), axis=-1))
        axis[self.a.data < -1e-6] = -axis[self.a.data < -1e-6]
        axis[axis.norm.data == 0] = Vector3d.zvector() * np.sign(self.a[axis.norm.data == 0].data)
        axis.data = axis.data / axis.norm.data[..., np.newaxis]
        return axis

    @property
    def angle(self):
        """Scalar : the angle of rotation."""
        return Scalar(2 * np.nan_to_num(np.arccos(np.abs(self.a.data))))

    @classmethod
    def random(cls, shape=(1,)):
        """Uniformly distributed rotations.

        Parameters
        ----------
        shape : int or tuple of int, optional
            The shape of the required object.

        """
        shape = (shape,) if isinstance(shape, int) else shape
        n = int(np.prod(shape))
        rotations = []
        while len(rotations) < n:
            r = np.random.uniform(-1, 1, (3*n, cls.dim))
            r2 = np.sum(np.square(r), axis=1)
            r = r[np.logical_and(1e-9 ** 2 < r2, r2 <= 1)]
            rotations += list(r)
        return cls(np.array(rotations[:n])).reshape(*shape)

    @classmethod
    def random_vonmises(cls, shape=(1,), alpha=1., reference=(1, 0, 0, 0), eps=1e-6):
        """Random rotations with a simplified Von Mises-Fisher distribution.

        Parameters
        ----------
        shape : int or tuple of int, optional
            The shape of the required object.
        alpha : float
            Parameter for the VM-F distribution. Lower values lead to "looser"
            distributions.
        reference : Rotation
            The center of the distribution.
        eps : float
            A small fixed variable.

        """
        shape = (shape,) if isinstance(shape, int) else shape
        reference = Rotation(reference)
        n = int(np.prod(shape))
        sample_size = int(alpha) * n
        rotations = []
        f_max = von_mises(reference, alpha, reference)
        while len(rotations) < n:
            rotation = cls.random(sample_size)
            f = von_mises(rotation, alpha, reference)
            x = np.random.rand(sample_size)
            rotation = rotation[x * f_max < f]
            rotations += list(rotation)
        return cls.stack(rotations[:n]).reshape(*shape)

    @property
    def antipodal(self):
        """Rotation : this and antipodally equivalent rotations."""
        r = self.__class__(np.stack([self.data, -self.data], axis=0))
        r.improper = self.improper
        return r


def von_mises(x, alpha, reference=Rotation((1, 0, 0, 0))):
    """A vastly simplified Von Mises-Fisher distribution calculation.

    Parameters
    ----------
    x : Rotation
    alpha : float
        Lower values of alpha lead to "looser" distributions.
    reference : Rotation

    Notes
    -----
    This simplified version of the distribution is calculated using

    .. math:: \\frac{\\exp\\left(2\\alpha\\cos\\left(\\omega\\right)\\right)}{_0F_1\\left(\\frac{N}{2}, \\alpha^2\\right)}

    where :math:`\omega` is the angle between orientations and :math:`N` is the
    number of relevant dimensions, in this case 3.

    Returns
    -------
    ndarray

    """
    angle = x.angle_with(reference)
    return np.exp(2 * alpha * np.cos(angle.data)) / hyp0f1(1.5, alpha**2)