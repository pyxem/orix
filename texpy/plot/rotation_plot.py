from matplotlib import projections
from mpl_toolkits.mplot3d import Axes3D
from texpy.vector.neo_euler import Rodrigues, AxAngle


class RotationPlot(Axes3D):

    name = None
    transformation_class = None

    def scatter(self, xs, *args, **kwargs):
        x, y, z = self.transform(xs)
        super().scatter(x, y, z, *args, **kwargs)

    def plot(self, xs, *args, **kwargs):
        x, y, z = self.transform(xs)
        super().plot(x, y, z, *args, **kwargs)

    def transform(self, xs):
        from texpy.quaternion.rotation import Rotation
        if isinstance(xs, Rotation):
            transformed = self.transformation_class.from_rotation(xs)
        else:
            transformed = self.transformation_class(xs)
        x, y, z = transformed.x.data, transformed.y.data, transformed.z.data
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

