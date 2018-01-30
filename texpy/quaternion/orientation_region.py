import itertools

import numpy as np

from texpy.quaternion.quaternion import Quaternion
from texpy.quaternion.rotation import Rotation
from texpy.vector.vector3d import Vector3d
from texpy.vector.axangle import AxAngle


class OrientationRegion:

    normals = None
    vertices = None
    faces = None
    symmetry = None

    @classmethod
    def from_symmetry(cls, symmetry):

        rotations = Rotation(symmetry.data)
        rotations.improper = symmetry.improper
        normals = cls.get_normals(rotations)
        orientation_region = cls()
        orientation_region.normals = normals
        orientation_region.symmetry = symmetry
        return orientation_region.cleanup()

    @classmethod
    def get_normals(cls, rotations):
        rotations = rotations[~rotations.improper].unique()
        rotations = rotations[~np.isclose(rotations.angle, 0)].unique()
        # Unique elements found manually to allow index return
        if len(rotations.axis.data):
            axes, indices = np.unique(rotations.axis.data.round(9), axis=0,
                                      return_inverse=True)
            axes = np.concatenate((axes, -axes), axis=-2)
            angles = np.array(
                [min(rotations.angle[indices == index]) for index in
                 set(indices)])
            angles = np.concatenate((angles / 2, np.pi - angles / 2), axis=-1)
            normals = AxAngle.from_axes_angles(axes, angles).to_rotation()
        else:
            normals = Rotation(np.zeros((0, 4)))
        return normals

    def cleanup(self):
        greater_than_pi = self.normals.angle > np.pi - 1e-3
        if not np.any(greater_than_pi):
            normals = self.normals.unique()
            reciprocal = normals * AxAngle.from_axes_angles(
                -normals.axis, np.pi * np.ones(normals.shape)).to_rotation()
            inside = OrientationRegion.is_inside(
                normals.to_quaternion(),
                -reciprocal.to_quaternion()
            )
            if len(inside):
                self.normals = normals[inside]
        else:
            orientation_region_full = OrientationRegion()
            orientation_region_full.normals = self.normals[~greater_than_pi]
            orientation_region_full.cleanup()
            ind = self.contains(orientation_region_full.vertices)
            face_indices = orientation_region_full.face_indices()
            inside = []
            for f in face_indices:
                if len(f):
                    inside.append(np.any(ind[f]))
                else:
                    inside.append(False)
            inside = np.array(inside)
            normals = orientation_region_full.normals
            if len(inside):
                normals = normals[inside]
            self.normals = Rotation(np.concatenate([
                normals.data,
                self.normals[greater_than_pi].data,
            ]))
        self.vertices = self.calculate_vertices(self.normals)
        return self

    @staticmethod
    def calculate_vertices(normals):
        normals = Quaternion(normals)
        if normals.size == 0:
            return Rotation(np.zeros((0, 4)))
        if normals.size == 2 or normals.size == 1:
            vertices = Rotation([
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0]
            ])
            return vertices
        vertices_all = Rotation.stack([Quaternion.triple_cross(f[0], f[1], f[2]) for f in itertools.combinations(normals, 3)])
        vertices_all = vertices_all[~np.any(np.isnan(vertices_all.data), axis=-1)]
        vertices_all = vertices_all.unique().to_quaternion()
        vertices = vertices_all[OrientationRegion.is_inside(normals, vertices_all)]
        switch180 = np.isclose(vertices.angle, np.pi)
        vertices180 = vertices[switch180]
        vertices180 = Quaternion.stack([vertices180, AxAngle.from_axes_angles(vertices180.axis, -vertices180.angle).to_rotation().to_quaternion()])
        axis_sector = OrientationRegion.calculate_axis_sector(normals)
        inside = axis_sector.contains(vertices180.axis)
        if inside.size > switch180.size:
            vertices = vertices180[inside].flatten()
        else:
            vertices[switch180] = vertices180[inside]
        return vertices.unique()

    def face_indices(self):
        if self.vertices is not None:
            face_indices = [
                np.where(np.isclose(normal.dot(self.vertices), 0))[0]
                for normal in self.normals]
            for i, fi in enumerate(face_indices):
                if len(fi) < 3:
                    face_indices[i] = np.array([])
            return face_indices

    @staticmethod
    def is_inside(normals, quaternions):
        normals = Quaternion(normals)
        cosines = normals.dot_outer(quaternions)
        inside1 = cosines < 1e-6
        inside2 = cosines > -1e-6
        axes = tuple(range(len(normals.shape)))
        inside = np.logical_or(np.all(inside1, axis=axes),
                             np.all(inside2, axis=axes))
        return inside

    def contains(self, quaternions):
        return OrientationRegion.is_inside(self.normals, quaternions)

    # def max_rotation(self, vector):
    #
    #     q = (vector * (1 / self.normals.to_rodrigues().dot_outer(vector).max(axis=0))[:, np.newaxis]).data
    #     return q

    def project(self, quaternions):
        """Returns quaternions projected into the fundamental region.

        Parameters
        ----------
        quaternions : Quaternion

        Returns
        -------
        Quaternion

        """
        best_transforms = self.symmetry.dot_outer(quaternions).argmax(axis=0)
        quaternions_inside = quaternions * ~self.symmetry[best_transforms]
        return quaternions_inside

    @staticmethod
    def calculate_axis_sector(normals):
        from texpy.vector.spherical_region import SphericalRegion
        n = normals.axis
        ind = normals.angle > np.pi - 1e-3
        spherical_region = SphericalRegion()
        spherical_region.normals = n[ind]
        return spherical_region

    def axis_sector(self):
        return self.calculate_axis_sector(self.normals)


class MisorientationRegion(OrientationRegion):

    @classmethod
    def from_symmetry(cls, symmetry_1, symmetry_2):
        misorientation_region = cls()
        from texpy.quaternion.symmetry import Symmetry
        rotations = Rotation(symmetry_2).outer(Rotation(symmetry_1))

        disjoint = Symmetry.disjoint(symmetry_1, symmetry_2)
        spherical_region = disjoint.calculate_fundamental_sector()
        angles = [np.pi - 1e-6] * spherical_region.normals.size
        zero_order_symmetries = AxAngle.from_axes_angles(
            spherical_region.normals, angles).to_rotation()

        normals = cls.get_normals(rotations)

        normals = Rotation(
            np.concatenate((normals.data, zero_order_symmetries.data), axis=-2))
        misorientation_region.normals = normals
        misorientation_region.symmetry = (symmetry_1, symmetry_2)
        return misorientation_region.cleanup()

    def project(self, transformations):
        """Returns transformations projected into the fundamental region.

        Parameters
        ----------
        transformations : Quaternion

        Returns
        -------
        Quaternion


        """
        t = transformations
        s1, s2 = self.symmetry
        # First, symmetrise the transformations
        s, index = ((~s2.to_quaternion()).outer(~s1.to_quaternion())).unique(return_index=True)
        s = s.numerical_sort()
        s2_index, s1_index = np.unravel_index(index, (s2.size, s1.size), order='F')
        s12_index = np.argmax(np.abs(t.dot_outer(s)), axis=-1)
        s1_index = s1_index[s12_index]
        s2_index = s2_index[s12_index]
        dat = s2[s2_index] * t * s1[s1_index]
        return dat


class Face(Quaternion):

    def __init__(self, normal, vertices):
        super(Face, self).__init__(normal)
        self.vertices = vertices
        if vertices is not None:
            v = vertices.to_rodrigues()
            diff = v - v.mean
            sign = np.sign(v.mean.dot(diff.cross(diff[0])))
            sign[np.isclose(sign, 0)] = 1
            angles = Vector3d.angle(diff, diff[0])
            ind = np.argsort(angles * sign)
            self.vertices = Rotation(vertices.data[ind])
        self.normal = Quaternion(normal)

    # def plot(self, ax=None):
    #     ax = plt.figure(figsize=(8, 8)).add_subplot(111, projection='3d',
    #                                                 aspect='equal') if ax is None else ax
    #     if self.vertices is not None and self.vertices.size > 2:
    #         poly = Poly3DCollection([self.vertices.to_rodrigues().data],
    #                                 edgecolor='k', alpha=0.2)
    #         poly.set_facecolor('C0')
    #         ax.add_collection(poly)
    #     else:
    #         Rotation(self.normal).to_rodrigues().plot_points(ax)
    #     return ax

