import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from texpy.plot.object3d_plot import Object3dPlot


class VectorPlot(Object3dPlot):

    def __init__(self, vector, ax=None):
        ax = plt.figure(figsize=(6, 6)).add_subplot(111, projection='3d', aspect='equal') \
            if ax is None else ax
        super(VectorPlot, self).__init__(vector, ax)

    def plot_1d(self, data, **kwargs):
        x, y, z = data[..., 0], data[..., 1], data[..., 2]
        self.ax.scatter(x, y, z, **kwargs)

    def plot_2d(self, data, **kwargs):
        self.plot_1d(data, **kwargs)

