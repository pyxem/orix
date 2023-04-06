r"""
===================
Combining rotations
===================

This example demonstrates how to combine two rotations :math:`r_A` and
:math:`r_B`, i.e. from left to right like so

.. math::

    r_{AB} = r_A \cdot r_B.

This order follows from the convention of passive rotations chosen in
orix which follows :cite:`rowenhorst2015consistent`.

To convince ourselves that this order is correct, we will reproduce the
example given by Rowenhorst and co-workers in section 4.2.2 in the above
mentioned paper. We want to rotate a vector :math:`(0, 0, z)` by two
rotations: rotation :math:`A` by :math:`120^{\circ}` around
:math:`[1 1 1]`, and rotation :math:`B` by :math:`180^{\circ}` around
:math:`[1 1 0]`; rotation :math:`A` will be carried out first, followed
by rotation :math:`B`.
"""

import matplotlib.pyplot as plt

from orix import plot
from orix.quaternion import Rotation
from orix.vector import Vector3d

plt.rcParams.update({"font.size": 12, "grid.alpha": 0.5})

rA = Rotation.from_axes_angles([1, 1, 1], -120, degrees=True)
rB = Rotation.from_axes_angles([1, 1, 0], -180, degrees=True)
rAB = rA * rB

# Compare with quaternions and orientation matrices from section 4.2.2
# in Rowenhorst et al. (2015)
r_all = Rotation.stack((rA, rB, rAB)).squeeze()
print("rA, rB and rAB:\n* As quaternions:\n", r_all)
print("* As orientation matrices:\n", r_all.to_matrix().squeeze().round(10))

v_start = Vector3d.zvector()
v_end = rAB * v_start
print(
    "Point rotated by rAB:\n",
    v_start.data.squeeze().tolist(),
    "->",
    v_end.data.squeeze().round(10).tolist(),
)

# Illustrate the steps of the rotation by plotting the vector before
# (red), during (green) and after (blue) the rotation and the rotation
# paths (first: cyan; second: magenta)
v_intermediate = rB * v_start

v_si_path = Vector3d.get_path(Vector3d.stack((v_start, v_intermediate)))
v_sie_path = Vector3d.get_path(Vector3d.stack((v_intermediate, v_end)))

fig = plt.figure()
ax0 = fig.add_subplot(121, projection="stereographic", hemisphere="upper")
ax1 = fig.add_subplot(122, projection="stereographic", hemisphere="lower")
ax0.stereographic_grid(), ax1.stereographic_grid()
Vector3d.stack((v_start, v_intermediate, v_end)).scatter(
    figure=fig,
    s=50,
    c=["r", "g", "b"],
    axes_labels=[r"$\mathbf{e}_x$", r"$\mathbf{e}_y$"],
)
ax0.plot(v_si_path, color="c"), ax1.plot(v_si_path, color="c")
ax0.plot(v_sie_path, color="m"), ax1.plot(v_sie_path, color="m")
rA.axis.scatter(figure=fig, c="orange")
rB.axis.scatter(figure=fig, c="k")
text_kw = dict(bbox=dict(alpha=0.5, fc="w", boxstyle="round,pad=0.1"), ha="right")
ax0.text(v_start, s="Start", **text_kw)
ax1.text(v_intermediate, s="Intermediate", **text_kw)
ax1.text(v_end, s="End", **text_kw)
ax1.text(rA.axis, s="Axis rA", **text_kw)
ax0.text(rB.axis, s="Axis rB", **text_kw)
