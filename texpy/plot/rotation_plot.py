import matplotlib.pyplot as plt
from texpy.plot.vector3d_plot import VectorPlot


class RotationPlot(VectorPlot):

    def __init__(self, rotation, ax=None):
        vector = rotation.to_axangle()
        super(RotationPlot, self).__init__(vector, ax)


def plot_pole_figure(rotation, **kwargs):
    _, (ax_north, ax_south) = plt.subplots(1, 2)
    ax_south.set_aspect('equal')
    ax_north.set_aspect('equal')
    x, y, z = rotation.axis.xyz
    north = z >= 0
    south = z < 0
    xn, yn = (x[north] / (1 + z[north]), y[north] / (1 + z[north]))
    xs, ys = (x[south] / (1 - z[south]), y[south] / (1 - z[south]))
    ax_north.scatter(xn, yn, **kwargs)
    ax_south.scatter(xs, ys, **kwargs)
    return ax_north, ax_south
