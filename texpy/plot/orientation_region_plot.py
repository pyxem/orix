import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from texpy.vector.neo_euler import Rodrigues
from texpy.quaternion.rotation import Rotation


class OrientationRegionPlot:

    def __init__(self, orientation_region, ax=None, figsize=(6, 6)):

        if ax is None:
            self.ax = plt.figure(figsize=figsize).add_subplot(111, projection='3d', aspect='equal')
        else:
            self.ax = ax
        self.orientation_region = orientation_region

    def draw(self, **kwargs):
        vertices_in_faces = np.isclose(
            self.orientation_region.normals.dot_outer(
                self.orientation_region.vertices).data, 0)
        faces = []
        for f in vertices_in_faces:
            faces.append(self.orientation_region.vertices[f].to_rotation())
        faces_sorted = []
        for face in faces:
            face_rodrigues = face.to_rodrigues()
            center = Rodrigues(face_rodrigues.data.mean(axis=0))
            disp = face_rodrigues - center
            sign = np.sign(disp.cross(disp[0]).unit.dot(center.unit).data)
            sign[np.isclose(sign, 0)] = 1
            face_sorted = face[np.argsort(sign * disp.angle_with(disp[0]).data)]
            face_sorted = Rotation(np.concatenate(
                (face_sorted.data, face_sorted[0].data)))
            faces_sorted.append(face_sorted)
        for face in faces_sorted:
            self.ax.plot(*face.to_rotation().to_axangle().xyz, 'k-', linewidth=0.5, **kwargs)
        lim = np.abs(np.concatenate([
            self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim()
        ])).max() * 1.1
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)
        plt.tight_layout()
        return self.ax



