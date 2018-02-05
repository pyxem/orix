import matplotlib.pyplot as plt


class Object3dPlot:

    def __init__(self, obj, ax=None):

        self.obj = obj
        ax = plt.figure(figsize=(6, 6)).add_subplot(111) if ax is None else ax
        self.ax = ax
        if obj.data_dim == 1:
            self.plot_function = self.plot_1d
        elif obj.data_dim == 2:
            self.plot_function = self.plot_2d
        else:
            raise NotImplementedError(
                'Plotting data with shape ({}) not supported'.format(
                    obj.shape))

    def draw(self, **kwargs):
        self.plot_function(self.obj.data, **kwargs)
        return self.ax

    def plot_1d(self, data, **kwargs):
        raise NotImplementedError

    def plot_2d(self, data, **kwargs):
        raise NotImplementedError
