r"""
=====================================================
Rotating z-vector to high-symmetry crystal directions
=====================================================

This example shows how to sample high-symmetry crystal directions
:math:`\mathbf{t} = [u, v, w]` (or zone axes) using
:meth:`orix.vector.Miller.from_highest_indices`.
We will also return the rotations :math:`R` which rotate
:math:`\mathbf{v_z} = (0, 0, 1)` to :math:`\mathbf{t}`.

We do the following to obtain the high-symmetry crystal directions:

1. Select a point group, here :math:`S = mmm`.
2. Sample all directions :math:`\mathbf{t_i}` with indices of -1, 0, and 1
3. Project :math:`\mathbf{t_i}` to the fundamental sector of the Laue group of :math:`S`
   (which in this case is itself)
4. Discard symmetrically equivalent and other duplicate crystal directions.
   Vectors such as [001] and [002] are considered equal after we make them unit vectors.
5. Round the vector indices to the closest smallest integer (below a default of 20).

The rotations :math:`R` can be useful e.g. when simulating diffraction patterns from
crystals with one of the high-symmetry zone axes :math:`\mathbf{t}` aligned along the
beam path.
"""

from orix.crystal_map import Phase
from orix.quaternion import Rotation
from orix.vector import Miller

phase = Phase(point_group="mmm")
t = Miller.from_highest_indices(phase, uvw=[1, 1, 1])
t = t.in_fundamental_sector()
t = t.unit.unique(use_symmetry=True).round()
print(t)

########################################################################
# Get the rotations that rotate :math:`\mathbf{v_z}` to these crystal
# directions
vz = Miller(uvw=[0, 0, 1], phase=t.phase)
R = Rotation.identity(t.size)
for i, t_i in enumerate(t):
    R[i] = Rotation.from_align_vectors(t_i, vz)
print(R)

########################################################################
# Plot the crystal directions within the fundamental sector of Laue
# group :math:`mmm`

fig = t.scatter(
    vector_labels=[str(vi).replace(".", "") for vi in t.coordinates],
    text_kwargs={
        "size": 15,
        "offset": (0, 0.03),
        "bbox": {"fc": "w", "pad": 2, "alpha": 0.75},
    },
    return_figure=True,
)
fig.axes[0].restrict_to_sector(t.phase.point_group.fundamental_sector)
