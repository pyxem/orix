from texpy.plot.object3d_plot import Object3dPlot


class ScalarPlot(Object3dPlot):

    def __init__(self, scalar, ax=None):
        super(ScalarPlot, self).__init__(scalar, ax)
        self.plot_1d = self.ax.plot
        self.plot_2d = self.ax.imshow