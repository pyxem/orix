r"""
=======================================
Misorientation from aligning directions
=======================================

This example demonstrates how to use
:meth:`~orix.quaternion.Misorientation.from_align_vectors` to estimate a misorientation
:math:`M` from two sets of aligned crystal directions :math:`t_{1,2}`, one set in each
crystal reference frame :math:`(e_1, e_2, e_3)`.
"""

from diffpy.structure import Lattice, Structure
import matplotlib.pyplot as plt
import numpy as np

from orix.crystal_map import Phase
from orix.quaternion import Misorientation, Orientation
from orix.vector import Miller

plt.rcParams.update({"figure.figsize": (5, 5), "lines.markersize": 8})

# Specify a cubic and an hexagonal crystal structures and symmetries
phase1 = Phase(
    point_group="m-3m",
    structure=Structure(lattice=Lattice(1, 1, 1, 90, 90, 90)),
)
phase2 = Phase(
    point_group="6/mmm",
    structure=Structure(lattice=Lattice(1, 1, 2, 90, 90, 120)),
)

# Specify one orientation per crystal
O_cubic = Orientation.from_axes_angles(
    [1, 1, 1], -60, symmetry=phase1.point_group, degrees=True
)
O_hex = Orientation.from_axes_angles(
    [1, 3, 2], -90, symmetry=phase2.point_group, degrees=True
)

# Get the reference misorientation (goal). Misorientations are obtained
# from the right, so: crystal 1 -> sample -> crystal 2
M_ref = Misorientation(O_hex * ~O_cubic, symmetry=(O_cubic.symmetry, O_hex.symmetry))

# Specify two directions in the first crystal
t_cubic = Miller(uvw=[[1, 1, 1], [0, 0, 1]], phase=phase1)

# Express the same directions with respect to the second crystal [in the
# Cartesian reference frame, (e1, e2, e3)]
v_hex = Miller(xyz=(M_ref * t_cubic).data, phase=phase2)

# Add some randomness to the second crystal directions (0 error
# magnitude gives exact result)
error_magnitude = 0.1
v_err = Miller(xyz=np.random.normal(0, error_magnitude, 3), phase=phase2)
v_hex_err = Miller(xyz=(v_hex + v_err).data, phase=phase2)
angle_err = v_hex_err.angle_with(v_hex, degrees=True).mean()
print("Vector angular deviation [deg]: ", angle_err)

# Get the misorientation that aligns the directions in the first crystal
# with those in the second crystal
M_new, err = Misorientation.from_align_vectors(v_hex_err, t_cubic, return_rmsd=True)
print("Error distance: ", err)

# Plot the two directions in the (unrotated) first crystal's reference
# frame as open circles
fig = t_cubic.scatter(
    c="none",
    ec=["r", "b"],
    grid=True,
    axes_labels=["e1", "e2"],
    return_figure=True,
    figure_kwargs={"layout": "tight"},
)

# Plot the two directions in the second crystal with respect to the
# first crystal's axes
(~M_ref * v_hex_err).scatter(c=["r", "b"], figure=fig)
