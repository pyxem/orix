"""An orientation region is some subset of the complete space of orientations.

The complete orientation space represents every possible orientation of an
object. The whole space is not always needed, for example if the orientation
of an object is constrained or (most commonly) if the object is symmetrical. In
this case, the space can be segmented using sets of Rotations representing
boundaries in the space. This is clearest in the Rodrigues parametrisation,
where the boundaries are planes, such as the example here: the asymmetric
domain of an adjusted 432 symmetry.

.. image:: /_static/img/orientation-region-Oq.png
   :width: 200px
   :alt: Boundaries of an orientation region in Rodrigues space.
   :align: center

Rotations or orientations can be inside or outside of an orientation region.

"""

import itertools

import numpy as np
from texpy.quaternion import Quaternion
from texpy.quaternion.rotation import Rotation
from texpy.quaternion.symmetry import C1, get_distinguished_points
from texpy.vector.neo_euler import Rodrigues, AxAngle


def _get_large_cell_normals(s1, s2):
    dp = get_distinguished_points(s1, s2)
    normals = Rodrigues.zero(dp.shape + (2,))
    planes1 = dp.axis * np.tan(dp.angle.data / 4)
    planes2 = -dp.axis * np.tan(dp.angle.data / 4) ** -1
    planes2.data[np.isnan(planes2.data)] = 0
    normals[:, 0] = planes1
    normals[:, 1] = planes2
    normals = Rotation.from_neo_euler(normals).flatten().unique(antipodal=False)
    return normals


def get_proper_groups(Gl, Gr):
    """Return the appropriate groups for the asymmetric domain calculation.

    Parameters
    ----------
    Gl, Gr : Symmetry

    Returns
    -------
    Gl, Gr : Symmetry
        The proper subgroup(s) or proper inversion subgroup(s) as appropriate.

    Raises
    ------
    NotImplementedError
        If both groups are improper, special consideration is needed which is
        not yet implemented in texpy.

    """
    if Gl.is_proper and Gr.is_proper:
        return Gl, Gr
    elif Gl.is_proper and not Gr.is_proper:
        return Gl, Gr.proper_subgroup
    elif not Gl.is_proper and Gr.is_proper:
        return Gl.proper_subgroup, Gr
    else:
        if Gl.contains_inversion and Gr.contains_inversion:
            return Gl.proper_subgroup, Gr.proper_subgroup
        elif Gl.contains_inversion and not Gr.contains_inversion:
            return Gl.proper_subgroup, Gr.proper_inversion_subgroup
        elif not Gl.contains_inversion and Gr.contains_inversion:
            return Gl.proper_inversion_subgroup, Gr.proper_subgroup
        else:
            raise NotImplementedError('Both groups are improper, '
                                      'and do not contain inversion.')


class OrientationRegion(Rotation):
    """A set of :obj:`Rotation`s which are the normals of an orientation region.

    """

    @classmethod
    def from_symmetry(cls, s1, s2=C1):
        s1, s2 = get_proper_groups(s1, s2)
        large_cell_normals = _get_large_cell_normals(s1, s2)
        disjoint = s1 & s2
        fundamental_sector = disjoint.fundamental_sector()
        fundamental_sector_normals = Rotation.from_neo_euler(
            AxAngle.from_axes_angles(fundamental_sector, np.pi))
        normals = Rotation(np.concatenate(
            [large_cell_normals.data, fundamental_sector_normals.data]))
        return cls(normals)

    def vertices(self):
        normal_combinations = list(itertools.combinations(self, 3))
        c1, c2, c3 = zip(*normal_combinations)
        c1, c2, c3 = Rotation.stack(c1).flatten(), Rotation.stack(
            c2).flatten(), Rotation.stack(c3).flatten()
        v = Rotation.triple_cross(c1, c2, c3)
        v = v[~np.any(np.isnan(v.data), axis=-1)]
        v = v[v < self].unique()
        surface = np.any(np.isclose(v.dot_outer(self).data, 0), axis=1)
        return v[surface]

    def faces(self):
        normals = Rotation(self)
        vertices = self.vertices()
        faces = []
        for n in normals:
            faces.append(vertices[np.isclose(vertices.dot(n).data, 0)])
        faces = [f for f in faces if f.size > 2]
        return faces

    def __gt__(self, other):
        c = Quaternion(self).dot_outer(Quaternion(other))
        inside = np.all(c > -1e-9, axis=0) | np.all(c < 1e-9, axis=0)
        return inside


