import numpy as np


class S1Grid:

    points = None

    def __init__(self, points):
        self.points = np.array(points)

    @property
    def minimum(self):
        return np.min(self.points)

    @property
    def maximum(self):
        return np.max(self.points)