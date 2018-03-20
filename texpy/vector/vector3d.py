import numpy as np

from texpy.base.object3d import Object3d, check
from texpy.scalar.scalar import Scalar
from texpy.plot.vector3d_plot import VectorPlot


def check_vector(obj):
    return check(obj, Vector3d)


class Vector3d(Object3d):
    """Three-dimensional Vector objects.

    Vectors typically represent positions or motion in 3d space.

    Attributes
    ----------
    data : ndarray
        The numpy array containing the vector data.

    """

    dim = 3
    data = None
    plot_type = VectorPlot

    def __neg__(self):
        return self.__class__(-self.data)

    def __add__(self, other):
        if isinstance(other, Vector3d):
            return self.__class__(self.data + other.data)
        elif isinstance(other, Scalar):
            return self.__class__(self.data + other.data[..., np.newaxis])
        elif isinstance(other, (int, float)):
            return self.__class__(self.data + other)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(self.data + other[..., np.newaxis])
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Vector3d):
            return self.__class__(other.data + self.data)
        elif isinstance(other, Scalar):
            return self.__class__(other.data[..., np.newaxis] + self.data)
        elif isinstance(other, (int, float)):
            return self.__class__(other + self.data)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(other[..., np.newaxis] + self.data)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector3d):
            return self.__class__(self.data - other.data)
        elif isinstance(other, Scalar):
            return self.__class__(self.data - other.data[..., np.newaxis])
        elif isinstance(other, (int, float)):
            return self.__class__(self.data - other)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(self.data - other[..., np.newaxis])
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Vector3d):
            return self.__class__(other.data - self.data)
        elif isinstance(other, Scalar):
            return self.__class__(other.data[..., np.newaxis] - self.data)
        elif isinstance(other, (int, float)):
            return self.__class__(other - self.data)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(other[..., np.newaxis] - self.data)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Vector3d):
            raise ValueError('Multiplying one vector with another is ambiguous. '
                             'Try `.dot` or `.cross` instead.')
        elif isinstance(other, Scalar):
            return self.__class__(self.data * other.data[..., np.newaxis])
        elif isinstance(other, (int, float)):
            return self.__class__(self.data * other)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(self.data * other[..., np.newaxis])
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Vector3d):
            raise ValueError('Multiplying one vector with another is ambiguous. '
                             'Try `.dot` or `.cross` instead.')
        elif isinstance(other, Scalar):
            return self.__class__(other.data[..., np.newaxis] * self.data)
        elif isinstance(other, (int, float)):
            return self.__class__(other * self.data)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(other[..., np.newaxis] * self.data)
        return NotImplemented

    def dot(self, other):
        """The dot product of a vector with another vector.

        Vectors must have compatible shape for broadcasting to work.

        Parameters
        ----------
        other : Vector3d

        Returns
        -------
        Scalar

        Examples
        --------
        >>> v = Vector3d((0, 0, 1.0))
        >>> w = Vector3d(((0, 0, 0.5), (0.4, 0.6, 0)))
        >>> v.dot(w)
        Scalar (2,)
        [ 0.5  0. ]
        >>> w.dot(v)
        Scalar (2,)
        [ 0.5  0. ]
        """
        if not isinstance(other, Vector3d):
            raise ValueError('{} is not a vector!'.format(other))
        return Scalar(np.sum(self.data * other.data, axis=-1))

    def dot_outer(self, other):
        """The outer dot product of a vector with another vector.

        The dot product for every combination of vectors in `self` and `other`
        is computed.

        Parameters
        ----------
        other : Vector3d

        Returns
        -------
        Scalar

        Examples
        --------
        >>> v = Vector3d(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)))  # shape = (2, )
        >>> w = Vector3d(((0.0, 0.0, 0.5), (0.4, 0.6, 0.0), (0.5, 0.5, 0.5)))  # shape = (3, )
        >>> v.dot_outer(w)
        Scalar (2, 3)
        [[ 0.5  0.   0.5]
         [ 0.   0.4  0.5]]
        >>> w.dot_outer(v)  # shape = (3, 2)
        Scalar (3, 2)
        [[ 0.5  0. ]
         [ 0.   0.4]
         [ 0.5  0.5]]

        """
        dots = np.tensordot(self.data, other.data, axes=(-1, -1))
        return Scalar(dots)

    def cross(self, other):
        """The cross product of a vector with another vector.

        Vectors must have compatible shape for broadcasting to work.

        Parameters
        ----------
        other : Vector3d

        Returns
        -------
        Vector3d
            The class of 'other' is preserved.

        Examples
        --------
        >>> v = Vector3d(((1, 0, 0), (-1, 0, 0)))
        >>> w = Vector3d((0, 1, 0))
        >>> v.cross(w)
        Vector3d (2,)
        [[ 0  0  1]
         [ 0  0 -1]]

        """
        return other.__class__(np.cross(self.data, other.data))

    @classmethod
    def from_polar(cls, theta, phi, r=1):
        """Creates a Vector3d object from polar data.

        Parameters
        ----------
        theta : array-like
            The polar angle, in radians.
        phi : array-like
            The azimuthal angle, in radians.
        r : array-like
            The radial distance. Defaults to 1 to produce unit vectors.

        Returns
        -------
        Vector3d

        """
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        z = np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        x = np.cos(phi) * np.sin(theta)
        return r * cls(np.stack((x, y, z), axis=-1))

    @classmethod
    def zero(cls, shape):
        """Returns zero vectors in the specified shape.

        Parameters
        ----------
        shape : tuple

        Returns
        -------
        Vector3d

        """
        return cls(np.zeros(shape + (cls.dim,)))

    @classmethod
    def xvector(cls):
        """Returns a single unit vector parallel to the x-direction.

        Returns
        -------
        Vector3d

        """
        return cls((1, 0, 0))

    @classmethod
    def yvector(cls):
        """Returns a single unit vector parallel to the y-direction.

        Returns
        -------
        Vector3d

        """
        return cls((0, 1, 0))

    @classmethod
    def zvector(cls):
        """Returns a single unit vector parallel to the z-direction.

        Returns
        -------
        Vector3d

        """
        return cls((0, 0, 1))

    @property
    def x(self):
        """Scalar : This vector's x data."""
        return Scalar(self.data[..., 0])

    @x.setter
    def x(self, value):
        self.data[..., 0] = value

    @property
    def y(self):
        """Scalar : This vector's y data."""
        return Scalar(self.data[..., 1])

    @y.setter
    def y(self, value):
        self.data[..., 1] = value

    @property
    def z(self):
        """Scalar : This vector's z data."""
        return Scalar(self.data[..., 2])

    @z.setter
    def z(self, value):
        self.data[..., 2] = value

    @property
    def xyz(self):
        """tuple of ndarray : This vector's components, useful for plotting."""
        return self.x.data, self.y.data, self.z.data

    def angle_with(self, other):
        """Calculate the angles between vectors in 'self' and 'other'

        Vectors must have compatible shapes for broadcasting to work.

        Parameters
        ----------
        other : Vector3d

        Returns
        -------
        Scalar
            The angle between the vectors, in radians.

        """
        cosines = np.round(self.dot(other).data / self.norm.data / other.norm.data, 9)
        return Scalar(np.arccos(cosines))

    def rotate(self, axis=None, angle=None):
        """Convenience function for rotating this vector.

        Shapes of 'axis' and 'angle' must be compatible with shape of this
        vector for broadcasting.

        Parameters
        ----------
        axis : array-like | Vector3d
            The axis of rotation. Defaults to the z-vector.
        angle : array-like
            The angle of rotation, in radians. Defaults to 0.

        Returns
        -------
        Vector3d

        Examples
        --------
        >>> from math import pi
        >>> v = Vector3d((0, 1, 0))
        >>> axis = Vector3d((0, 0, 1))
        >>> angles = [0, pi/4, pi/2, 3*pi/4, pi]
        >>> v.rotate(axis=axis, angle=angles)


        """
        from texpy.vector.axangle import AxAngle
        axis = Vector3d.zvector() if axis is None else axis
        angle = 0 if angle is None else angle
        q = AxAngle.from_axes_angles(axis, angle).to_rotation()
        return q * self
