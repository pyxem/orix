import numpy as np
from texpy.scalar.scalar import Scalar
from texpy.vector.neo_euler import NeoEuler


class Rodrigues(NeoEuler):

    @property
    def angle(self):
        return Scalar(np.arctan(self.norm.data) * 2)

