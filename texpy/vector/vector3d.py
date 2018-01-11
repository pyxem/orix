import numpy as np
import itertools

from texpy.object3d import Object3d, check_matching_type
# from texpy.plot.plot import Scatter3d


class Vector3d(Object3d):

    dim = 3
    data = None

    def __neg__(self):
        return self.__class__(-self.data)

    def __add__(self, other):
        if isinstance(other, Vector3d):
            return self.__class__(self.data + other.data)
        elif isinstance(other, (int, float)):
            return self.__class__(self.data + other)
        else:
            return self.data + other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Vector3d):
            return self.__class__(self.data - other.data)
        elif isinstance(other, (int, float)):
            return self.__class__(self.data - other)
        else:
            return self.data - other

    def __rsub__(self, other):
        if isinstance(other, Vector3d):
            return self.__class__(other.data - self.data)
        elif isinstance(other, (int, float)):
            return self.__class__(other - self.data)
        else:
            return other - self.data

    def __mul__(self, other):
        if isinstance(other, Vector3d):
            raise ValueError("Multiplying vectors is ambiguous. "
                             "Use 'dot' or 'cross'.")
        elif isinstance(other, np.ndarray):
            data = other[..., np.newaxis] * self.data
            c = self.__class__(data)
            return c
        elif isinstance(other, (float, int)):
            return self.__class__(self.data * other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, np.ndarray):
            return self.__class__(other[..., np.newaxis] * self.data)
        if isinstance(other, (float, int)):
            return self.__class__(self.data * other)
        return NotImplemented

    def dot(self, other):
        other = check_matching_type(self, other)
        return np.sum(self.data * other.data, axis=-1)

    def dot_outer(self, other):
        dots = np.tensordot(self.data, other.data, axes=(-1, -1))
        return dots

    def cross(self, other):
        other = check_matching_type(self, other)
        return self.__class__(np.cross(self.data, other.data))

    @classmethod
    def from_polar(cls, theta, phi, r=1):
        theta = np.atleast_1d(theta)
        phi = np.atleast_1d(phi)
        z = np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        x = np.cos(phi) * np.sin(theta)
        return r * cls(np.stack((x, y, z), axis=-1))

    @classmethod
    def zero(cls):
        return cls((0, 0, 0))

    @classmethod
    def xvector(cls):
        return cls((1, 0, 0))

    @classmethod
    def yvector(cls):
        return cls((0, 1, 0))

    @classmethod
    def zvector(cls):
        return cls((0, 0, 1))

    @property
    def x(self):
        return self.data[..., 0]

    @x.setter
    def x(self, value):
        self.data[..., 0] = value

    @property
    def y(self):
        return self.data[..., 1]

    @y.setter
    def y(self, value):
        self.data[..., 1] = value

    @property
    def z(self):
        return self.data[..., 2]

    @z.setter
    def z(self, value):
        self.data[..., 2] = value

    # @property
    # def arrows(self):
    #     return [Arrow3d((0, v[0]), (0, v[1]), (0, v[2]),
    #                     mutation_scale=10,
    #                     arrowstyle="wedge",
    #                     color='r',
    #                     shrinkA=0, shrinkB=0
    #                     ) for v in self.data]

    def angle_with(self, other):
        """Calculate the angles between all vectors in 'self' and 'other'"""
        other = check_matching_type(self, other)
        cosines = np.round(self.dot(other) / self.norm / other.norm, 9)
        return np.arccos(cosines)

    def rotate(self, q):
        from texpy.quaternion.rotation import Rotation
        from texpy.vector.axangle import AxAngle
        if isinstance(q, (float, int)):
            q = AxAngle.from_axes_angles(Vector3d.zvector(), q).to_rotation()
        q = Rotation(q)
        return q * self

    def plot_points(self, ax=None):
        x = self.x.flatten()
        y = self.y.flatten()
        z = self.z.flatten()
        scatter_plot = Scatter3d(x, y, z, ax)
        ax = scatter_plot.draw()
        return ax
    #
    # def plot_arrows(self, ax=None):
    #     ax = plt.figure().add_subplot(111, projection='3d',
    #                                   aspect='equal') if ax is None else ax
    #     length = np.abs(self.length).max()
    #     ax.set_xlim(-length, length)
    #     ax.set_ylim(-length, length)
    #     ax.set_zlim(-length, length)
    #     for arrow in self.arrows:
    #         ax.add_artist(arrow)
    #     return ax
    #
    # def plot_normals(self, ax=None):
    #     ax = plt.figure().add_subplot(111, projection='3d',
    #                                   aspect='equal') if ax is None else ax
    #     xx, yy = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
    #     d = -np.sum(self * self.length[:, np.newaxis] * self, axis=1)
    #
    #     # Plot z
    #     mask = np.logical_and(~np.isclose(self.z, 0), np.isclose(self.x, 0))
    #     z = (- self.x[mask] * xx[:, :, np.newaxis]
    #          - self.y[mask] * yy[:, :, np.newaxis]
    #          - d[mask]) * self.z[mask] ** -1
    #     for zt in z.T:
    #         ax.plot_surface(xx, yy, zt.T, alpha=0.25)
    #
    #     # Plot x
    #     mask = np.logical_and(~np.isclose(self.y, 0), np.isclose(self.z, 0))
    #     z = (- self.z[mask] * xx[:, :, np.newaxis]
    #          - self.x[mask] * yy[:, :, np.newaxis]
    #          - d[mask]) * self.y[mask] ** -1
    #     for zt in z.T:
    #         ax.plot_surface(zt.T, yy, xx, alpha=0.25)
    #
    #     # Plot y
    #     mask = np.logical_and(~np.isclose(self.x, 0), np.isclose(self.y, 0))
    #     z = (- self.y[mask] * xx[:, :, np.newaxis]
    #          - self.z[mask] * yy[:, :, np.newaxis]
    #          - d[mask]) * self.x[mask] ** -1
    #     for zt in z.T:
    #         ax.plot_surface(xx, zt.T, yy, alpha=0.25)
    #     return ax


# class Arrow3d(FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
#         self._verts3d = xs, ys, zs
#
#     def draw(self, renderer):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         FancyArrowPatch.draw(self, renderer)
