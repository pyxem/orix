import numpy as np

from .vector3d import Vector3d
from texpy.scalar.scalar import Scalar
from texpy.vector.neo_euler import NeoEuler


class AxAngle(NeoEuler):

    @property
    def angle(self):
        return Scalar(self.norm.data)

    @classmethod
    def from_axes_angles(cls, axes, angles):
        axes = Vector3d(axes).unit
        angles = np.array(angles)
        axangle_data = angles[..., np.newaxis] * axes.data
        return cls(axangle_data)

