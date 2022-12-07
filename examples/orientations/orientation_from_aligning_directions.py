"""
====================================
Orientation from aligning directions
====================================

This example demonstrates how to use
:meth:`~orix.quaternion.Orientation.from_align_vectors` to estimate an orientation from
two sets of aligned vectors, one set in the sample reference reference frame, the other
in the crystal reference frame.
"""

from diffpy.structure import Lattice, Structure
import numpy as np

from orix.crystal_map import Phase
from orix.quaternion import Orientation
from orix.vector import Miller, Vector3d

# Specify a crystal structure and symmetry
phase = Phase(
    point_group="6/mmm", structure=Structure(lattice=Lattice(1, 1, 2, 90, 90, 120))
)

# Define a reference orientation (goal)
ori_ref = Orientation.from_axes_angles([1, 1, 1], np.pi / 4, phase.point_group)

# Specify two crystal directions (any will do)
vec_c = Miller(uvw=[[2, 1, 1], [1, 3, 1]], phase=phase)

# Find out where these directions in the reference orientation (crystal)
# point in the sample reference frame
vec_r = Vector3d(~ori_ref * vec_c)

# Plot the reference orientation sample directions as empty circles
fig = vec_r.scatter(
    ec=["r", "b"],
    s=100,
    fc="none",
    grid=True,
    axes_labels=["X", "Y"],
    return_figure=True,
)
fig.tight_layout()

# Add some randomness to the sample directions (0 error magnitude gives
# exact result)
err_magnitude = 0.1
vec_err = Vector3d(np.random.normal(0, err_magnitude, 3))
vec_r_err = vec_r + vec_err
angle_err = np.rad2deg(vec_r_err.angle_with(vec_r)).mean()
print("Vector angle deviation [deg]: ", angle_err)

# Obtain the orientation which aligns the crystal directions with the
# sample directions
ori_new_r2c, err = Orientation.from_align_vectors(vec_c, vec_r_err, return_rmsd=True)
ori_new_c2r = ~ori_new_r2c
print("Error distance: ", err)

# Plot the crystal directions in the new orientation
vec_r2 = Vector3d(~ori_new_r2c * vec_c)
vec_r2.scatter(c=["r", "b"], figure=fig)
