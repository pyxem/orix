from texpy.plot.object3d_plot import Object3dPlot


class ScalarPlot(Object3dPlot):

    def __init__(self, scalar, ax=None, figsize=(6, 6)):
        super(ScalarPlot, self).__init__(scalar, ax, figsize)

    def plot_1d(self, data, **kwargs):
        self.ax.plot(data, **kwargs)

    def plot_2d(self, data, **kwargs):
        self.ax.imshow(data, **kwargs)