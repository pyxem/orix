"""Neo-Eulerian vectors parametrize rotations as vectors.

The rotation is specified by an axis of rotation and an angle. Different
neo-Eulerian vectors have different scaling functions applied to the angle
of rotation for different properties of the space. For example, the axis-angle
representation does not scale the angle of rotation, making it easy for direct
interpretation, whereas the Rodrigues representation applies a scaled tangent
function, such that any straight lines in Rodrigues space represent rotations
about a fixed axis.

"""
import abc

import numpy as np
from texpy.scalar import Scalar
from texpy.vector import Vector3d


class NeoEuler(Vector3d, abc.ABC):
    """Base class for neo-Eulerian vectors.

    """

    @classmethod
    @abc.abstractmethod
    def from_rotation(cls, rotation):
        """NeoEuler : Create a new vector from the given rotation."""
        pass

    @property
    @abc.abstractmethod
    def angle(self):
        """Scalar : the angle of rotation."""
        pass

    @property
    def axis(self):
        """Vector3d : the axis of rotation."""
        return Vector3d(self.unit)


class Homochoric(NeoEuler):
    """Equal-volume mapping of the unit quaternion hemisphere.

    The homochoric vector representing a rotation with rotation angle
    :math:`\\theta` has magnitude
    :math:`\\left[\\frac{3}{4}(\\theta - \\sin\\theta)\\right]^{\\frac{1}{3}}`.

    Notes
    -----
    The homochoric transformation has no analytical inverse.

    """

    @classmethod
    def from_rotation(cls, rotation):
        theta = rotation.angle.data
        n = rotation.axis
        magnitude = (0.75 * (theta - np.sin(theta)))**(1/3)
        return cls(n * magnitude)

    @property
    def angle(self):
        raise AttributeError("The angle of a homochoric vector cannot be determined analytically.")



class Rodrigues(NeoEuler):
    """In Rodrigues space, straight lines map to rotations about a fixed axis.

    The Rodrigues vector representing a rotation with rotation angle
    :math:`\\theta` has magnitude :math:`\\tan\\frac{\\theta}{2}`.

    """

    @classmethod
    def from_rotation(cls, rotation):
        a = np.float64(rotation.a.data)
        with np.errstate(divide='ignore', invalid='ignore'):
            data = np.stack((
                rotation.b.data / a,
                rotation.c.data / a,
                rotation.d.data / a
            ), axis=-1)
        data[np.isnan(data)] = 0
        r = cls(data)
        return r

    @property
    def angle(self):
        return Scalar(np.arctan(self.norm.data) * 2)


class AxAngle(NeoEuler):
    """The simplest neo-Eulerian representation.

    The Axis-Angle vector representing a rotation with rotation angle
    :math:`\\theta` has magnitude :math:`\\theta`

    """

    @classmethod
    def from_rotation(cls, rotation):
        return cls((rotation.axis * rotation.angle).data)

    @property
    def angle(self):
        return Scalar(self.norm.data)

    @property
    def axis(self):
        return Vector3d(self.unit)

    @classmethod
    def from_axes_angles(cls, axes, angles):
        """Create new AxAngle object explicitly from the given axes and angles.

        Parameters
        ----------
        axes : Vector3d or array_like
            The axis of rotation.
        angles : array_like
            The angle of rotation, in radians.

        Returns
        -------
        AxAngle

        """
        axes = Vector3d(axes).unit
        angles = np.array(angles)
        axangle_data = angles[..., np.newaxis] * axes.data
        return cls(axangle_data)