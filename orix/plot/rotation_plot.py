from matplotlib import projections
from mpl_toolkits.mplot3d import Axes3D
from orix.vector.neo_euler import Rodrigues, AxAngle


class RotationPlot(Axes3D):

    name = None
    transformation_class = None

    def scatter(self, xs, **kwargs):
        x, y, z = self.transform(xs)
        return super().scatter(x, y, z, **kwargs)

    def plot(self, xs, **kwargs):
        x, y, z = self.transform(xs)
        return super().plot(x, y, z, **kwargs)

    def plot_wireframe(self, xs, **kwargs):
        x, y, z = self.transform(xs)
        return super().plot_wireframe(x, y, z, **kwargs)

    def transform(self, xs):
        from orix.quaternion.rotation import Rotation
        if isinstance(xs, Rotation):
            transformed = self.transformation_class.from_rotation(xs.get_plot_data())
        else:
            transformed = self.transformation_class(xs)
        x, y, z = transformed.xyz
        return x, y, z


class RodriguesPlot(RotationPlot):
    """Plot rotations in a Rodrigues-Frank projection."""
    name = 'rodrigues'
    transformation_class = Rodrigues


class AxAnglePlot(RotationPlot):
    """Plot rotations in an Axes-Angle projection."""
    name = 'axangle'
    transformation_class = AxAngle


projections.register_projection(RodriguesPlot)
projections.register_projection(AxAnglePlot)

