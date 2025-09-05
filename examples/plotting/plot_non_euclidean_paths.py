r"""
========================================
Plot Paths Through Non-Euclidean Spaces
========================================

This example shows three variations on how 'from_path_ends' can be
used to plot paths between points in rotational and vector spaces.

This functionality is available in :class:`~orix.vector.Vector3d`,
:class:`~orix.quaternions.Rotation`,
:class:`~orix.quaternions.Orientation`,
and :class:`~orix.quaternions.Misorientation`.
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from orix.quaternion import Misorientation, Orientation, Rotation
from orix.quaternion.symmetry import D3, Oh
from orix.vector import Vector3d

fig = plt.figure(figsize=(4, 8))

# ========= #
# Example 1: Plotting a path of rotations with no symmetry in homochoric space
# ========= #
rots_along_path = Rotation(
    data=np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ]
    )
)
n_steps = 20
rotation_path = Rotation.from_path_ends(rots_along_path, steps=n_steps)
# create an Orientation loop using this path with no symmetry elements
ori_path = Orientation(rotation_path)
# plot the path in homochoric space
segment_colors = cm.inferno(np.linspace(0, 1, n_steps))

path_colors = np.vstack([segment_colors for x in range(rots_along_path.size - 1)])
ori_path.scatter(figure=fig, position=[3, 1, 1], marker=">", c=path_colors)
fig.axes[0].set_title(r"$90^\circ$ rotation around X, then Y")

# ========= #
# Example 2: Plotting the rotation of several orientations in m3m Rodrigues
# space around the z axis.
# ========= #
oris = Orientation(
    data=np.array(
        [
            [0.69, 0.24, 0.68, 0.01],
            [0.26, 0.59, 0.32, 0.7],
            [0.07, 0.17, 0.93, 0.31],
            [0.6, 0.03, 0.61, 0.52],
            [0.51, 0.38, 0.34, 0.69],
            [0.31, 0.86, 0.22, 0.35],
            [0.68, 0.67, 0.06, 0.31],
            [0.01, 0.12, 0.05, 0.99],
            [0.39, 0.45, 0.34, 0.72],
            [0.65, 0.59, 0.46, 0.15],
        ]
    ),
    symmetry=Oh,
).reduce()
# define a 20 degree rotation around the z axis
shift = Orientation.from_axes_angles([0, 0, 1], np.pi / 9)
segment_colors = cm.inferno(np.linspace(0, 1, 10))

ori_paths = []
for ori in oris:
    shifted = (shift * ori).reduce()
    to_from = Orientation.stack([ori, shifted]).flatten()
    ori_paths.append(Orientation.from_path_ends(to_from, steps=10))
# plot a path in roddrigues space with m-3m (cubic) symmetry.
ori_path = Orientation.stack(ori_paths).flatten()
ori_path.symmetry = Oh
ori_path.scatter(
    figure=fig,
    position=[3, 1, 2],
    marker=">",
    c=np.tile(segment_colors, [10, 1]),
    projection="rodrigues",
)
fig.axes[1].set_title(r"$20^{\circ}$ rotations around X-axis in m3m")

# ========= #
# Example 3: creating a customized Wulf Plotting the rotation of several orientations in m3m Rodrigues
# space around the z axis.
# ========= #


# plot vectors
ax_upper = plt.subplot(3, 1, 3, projection="stereographic", hemisphere="upper")
r90x = Rotation.from_axes_angles([1, -1, -1], [0, 60], degrees=True)
x_axis_points = r90x * Vector3d.xvector()
y_axis_points = r90x * Vector3d.yvector()
z_axis_points = r90x * Vector3d.zvector()

x_axis_path = Vector3d.from_path_ends(x_axis_points.unique())
y_axis_path = Vector3d.from_path_ends(y_axis_points.unique())
z_axis_path = Vector3d.from_path_ends(z_axis_points.unique())
cx = cm.Reds(np.linspace(0.1, 1, x_axis_path.size))
cy = cm.Greens(np.linspace(0.1, 1, y_axis_path.size))
cz = cm.Blues(np.linspace(0.1, 1, z_axis_path.size))

spx = ax_upper.scatter(x_axis_path, figure=fig, marker=">", c=cx, label="X")
spy = ax_upper.scatter(y_axis_path, figure=fig, marker=">", c=cy, label="Y")
spz = ax_upper.scatter(z_axis_path, figure=fig, marker=">", c=cz, label="Z")
ax_upper.legend(loc="lower center", ncols=3)

plt.tight_layout()
