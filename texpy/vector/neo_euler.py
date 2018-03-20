import numpy as np
from texpy.quaternion.rotation import Rotation
from texpy.vector.vector3d import Vector3d


class NeoEuler(Vector3d):

    def to_rotation(self):
        s = np.sin(self.angle.data / 2)
        a = np.cos(self.angle.data / 2)
        b = s * self.axis.x.data
        c = s * self.axis.y.data
        d = s * self.axis.z.data
        r = Rotation(np.stack([a, b, c, d], axis=-1))
        return r

    @property
    def axis(self):
        u = self.unit
        u[u.norm.data == 0] = Vector3d.xvector()
        return u
