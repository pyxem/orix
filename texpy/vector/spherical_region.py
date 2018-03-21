import numpy as np

from texpy.vector import Vector3d
from texpy.quaternion import Quaternion
from texpy.quaternion.rotation import Rotation


class SphericalRegion(Vector3d):
    """Defines a volume of a sphere based on bounding planes.

    """

    @classmethod
    def from_symmetry(cls, symmetry):
        """Calculates the fundamental sector of a sphere for 'symmetry'

        First, calculate the fundamental region according to mirror planes only.
        Then, divide this region according to the remaining rotational
        symmetry elements.

        Parameters
        ----------
        symmetry : Symmetry
            A symmetry object

        Returns
        -------
        SphericalRegion
            The fundamental sector of 'symmetry'

        """
        # Extract mirror planes
        mirror_planes = symmetry.axis[
            np.logical_and(symmetry.improper, np.abs(symmetry.angle.data) >= np.pi - 1e-6)]
        mirror_planes = mirror_planes[mirror_planes.z > 0 - 1e-6]
        mirror_planes = symmetry.outer(mirror_planes).unique()

        # Separate mirror planes parallel to z-axis
        zone = np.isclose(mirror_planes.dot(Vector3d.zvector()).data, 0)
        mirror_planes_z = mirror_planes[zone]
        mirror_planes_nz = mirror_planes[~zone]

        if np.any(np.isclose(symmetry.outer(Vector3d.zvector()).z.data, -1)) and np.any(symmetry.improper):
            mirror_planes_nz = Vector3d(np.concatenate((mirror_planes_nz.data, Vector3d.zvector().data))).unique()

        # We only ever need two mirror planes parallel to z-axis
        # The first should generally have normal along the y-axis, close to it.
        mirror_plane_z_1 = Vector3d.empty()
        if mirror_planes_z.size:
            order = np.lexsort([mirror_planes_z.y.data, mirror_planes_z.dot(-Vector3d.xvector()).data])
            mirror_plane_z_1 = mirror_planes_z[order[-1]]
        # The second will be that with the smallest angle to the first.
        mirror_plane_z_2 = Vector3d.empty()
        if mirror_planes_z.size:
            angles = mirror_planes_z.angle_with(mirror_plane_z_1).data
            angles[np.isclose(angles, 0)] = np.pi
            signs = np.sign(mirror_planes_z.cross(mirror_plane_z_1).z.data)
            order = np.lexsort([signs, angles])
            mirror_plane_z_2 = -mirror_planes_z[order[0]]
            # print(mirror_plane_z_1, mirror_plane_z_2)
        if np.all((mirror_plane_z_1 + mirror_plane_z_2).data < -1e-6):
            mirror_plane_z_1 = -mirror_plane_z_1
            mirror_plane_z_2 = -mirror_plane_z_2

        # Deal with the remaining mirror planes
        # The last will normally have a normal pointing *away* from the combined
        # normals of the first two planes, if they exist
        if mirror_planes_nz.size:
            if mirror_plane_z_1.size:
                if mirror_plane_z_2.size:
                    mplanenz = mirror_planes_nz[
                        mirror_planes_nz.dot(mirror_plane_z_1 + mirror_plane_z_2).data.argmin()]
                else:
                    mplanenz = mirror_planes_nz[mirror_planes_nz.dot(mirror_plane_z_1).data.argmin()]
            else:
                mplanenz = mirror_planes_nz[0]
        else:
            mplanenz = np.zeros((0, 3))

        # The supersector is now formed from the three mirror planes found
        supersector = SphericalRegion(np.concatenate([
                mirror_plane_z_1.data,
                mirror_plane_z_2.data,
                mplanenz.data
            ], axis=-2))

        # Deal with the rotations
        rotations = symmetry[~symmetry.improper]
        rotations = rotations.outer(rotations)
        r_inside = Rotation(Quaternion(rotations[supersector.contains(rotations.axis, edge=False)]).unique())
        r_inside = r_inside[r_inside.angle > 1e-6]

        # For every rotation inside the supersector, divide up the remaining
        # space. Mathematically, this is general for any symmetry elements,
        # but for a "nice" region the perpendicular vectors have to be picked.
        while r_inside.size:
            size = r_inside.size
            r = r_inside[np.isclose(r_inside.angle.data, r_inside.angle.data.min())]
            r = r[r.axis.angle_with(Vector3d.zvector()).data.argmin()]
            if np.isclose(r.axis.z.data, 1):
                perp = Vector3d.yvector()
            elif np.isclose(r.axis.x.data, 1):
                perp = Vector3d.zvector()
            elif np.isclose(r.axis.y.data, 1):
                perp = Vector3d.xvector()
            else:
                if np.isclose(np.sum(supersector.z.data), 0):
                    perp = Vector3d.yvector()
                else:
                    perp = Vector3d.zvector()
            n1 = r.axis.cross(perp).unit
            if n1.z < 0:
                n1 = -n1
            n2 = - (r * n1)
            supersector = SphericalRegion(np.concatenate(
                [supersector.data, n1.data, n2.data])).unique()
            inside = supersector.contains(r_inside.axis, edge=False)
            r_inside = r_inside[inside].unique()
            if r_inside.size == size:
                break
        return supersector

    @staticmethod
    def is_in_or_on(normals, vectors, comparator):
        normals = Vector3d(normals)
        vectors = Vector3d(vectors)
        cosines = normals.dot_outer(vectors)
        inside = np.all(cosines > comparator, axis=0)
        return inside

    def contains(self, vectors, edge=True):
        """Returns 'True' where 'vectors' is in or on the region.

        Parameters
        ----------
        vectors : Vector3d
        edge : bool
            If 'True' (default), the surface of the region is considered.
            If 'False', only internal vectors are considered.

        Returns
        -------
        ndarray

        """
        comparator = -1e-6 if edge is True else 1e-6
        return SphericalRegion.is_in_or_on(self, vectors, comparator)
