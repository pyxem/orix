"""
=======================================
Misorientation from aligning directions
=======================================

This example demonstrates how to use
:meth:`~orix.quaternion.Misorientation.from_align_vectors` to estimate a misorientation
from two sets of aligned crystal directions, one set in each crystal reference frame.
"""

from diffpy.structure import Lattice, Structure
import numpy as np

from orix.crystal_map import Phase
from orix.quaternion import Misorientation, Orientation
from orix.vector import Miller

# Specify two crystal structures and symmetries
phase1 = Phase(
    point_group="m-3m", structure=Structure(lattice=Lattice(1, 1, 1, 90, 90, 90))
)
phase2 = Phase(
    point_group="6/mmm", structure=Structure(lattice=Lattice(1, 1, 2, 90, 90, 120))
)

# Specify one orientation per crystal
ori_ref1 = Orientation.from_axes_angles(
    [1, 1, 1], np.pi / 3, symmetry=phase1.point_group
)
ori_ref2 = Orientation.from_axes_angles(
    [1, 3, 2], np.pi / 2, symmetry=phase2.point_group
)

# Get the reference misorientation (goal). Transformations are composed
# from the right, so: crystal 1 -> sample -> crystal 2
mori_ref = Misorientation(
    ori_ref2 * (~ori_ref1), symmetry=(ori_ref1.symmetry, ori_ref2.symmetry)
)

# Specify two directions in the first crystal
vec_c1 = Miller(uvw=[[1, 1, 1], [0, 0, 1]], phase=phase1)

# Express the same directions with respect to the second crystal
vec_c2 = Miller(xyz=(mori_ref * vec_c1).data, phase=phase2)

# Add some randomness to the second crystal directions (0 error
# magnitude gives exact result)
error_magnitude = 0.1
vec_err = Miller(xyz=np.random.normal(0, error_magnitude, 3), phase=phase2)
vec_c2_err = Miller(xyz=(vec_c2 + vec_err).data, phase=phase2)
angle_err = np.rad2deg(vec_c2_err.angle_with(vec_c2)).mean()
print("Vector angular deviation [deg]: ", angle_err)

# Get the misorientation that aligns the directions in the first crystal
# with those in the second crystal
mori_new, err = Misorientation.from_align_vectors(vec_c2_err, vec_c1, return_rmsd=True)
print("Error distance: ", err)

# Plot the two directions in the (unrotated) first crystal's reference
# frame as open circles
fig = vec_c1.scatter(
    ec=["r", "b"],
    s=100,
    fc="none",
    grid=True,
    axes_labels=["e1", "e2"],
    return_figure=True,
)
fig.tight_layout()

# Plot the two directions in the second crystal with respect to the
# first crystal's axes
(~mori_ref * vec_c2_err).scatter(c=["r", "b"], figure=fig)
