import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Scatter3d:

    def __init__(self, x, y, z, ax=None):
        self.x = x
        self.y = y
        self.z = z
        self.ax = plt.figure().add_subplot(
            111, projection='3d', aspect='equal') if ax is None else ax

    def draw(self):
        self.ax.plot(self.x, self.y, self.z, 'ko', markersize=1)
        return self.ax

