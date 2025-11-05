r"""
=========================================
Visualizing crystallographic paths
=========================================

This example shows how define and plot paths through either
rotation or vector space.This is akin to describing crystallographic
fiber textures in metallurgy, or the shortest arcs connecting points on
the surface of a unit sphere.

In both cases, "shortest" is defined as the route that minimizes the
movement required to transform from point to point, which is typically
not a stright line when plotted into a euclidean projection
(axis-angle, stereographic, etc.).

This functionality is available in :class:`~orix.vector.Vector3d`,
:class:`~orix.quaternions.Rotation`,
:class:`~orix.quaternions.Orientation`,
and :class:`~orix.quaternions.Misorientation`."""

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

from orix.plot import register_projections
from orix.plot.direction_color_keys import DirectionColorKeyTSL
from orix.quaternion import Orientation, OrientationRegion, Quaternion
from orix.quaternion.symmetry import C1, Oh
from orix.sampling import sample_S2
from orix.vector import Vector3d

plt.close("all")
register_projections()  # Register our custom Matplotlib projections
np.random.seed(2319)  # Create reproducible random data


fig = plt.figure(figsize=(6, 6))
n_steps = 30

# ============ #
# Example 1: Plotting multiple paths into a user defined axis

# This subplot shows several paths through the cubic (m3m) fundamental zone
# created by rotating 20 randomly chosen points 30 degrees around the z axis.
# these paths are drawn in rodrigues space, which is an equal-angle projection
# of rotation space. As such, notice how all lines tracing out axial rotations
# are straight, but lines starting closer to the center of the fundamental zone
# appear shorter.

# the sampe paths are then also plotted on an Inverse Pole Figure (IPF) plot.

rod_ax = fig.add_subplot(2, 2, 1, projection="rodrigues", proj_type="ortho")
ipf_ax = fig.add_subplot(2, 2, 2, projection="ipf", symmetry=Oh)

# 10 random orientations with the cubic m3m ('Oh' in the schoenflies notation)
# crystal symmetry.
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
)
# reduce them to their crystallographically identical representations
oris = oris.reduce()
# define a 20 degree rotation around the z axis
shift = Orientation.from_axes_angles([0, 0, 1], 30, degrees=True)
# for each orientation, calculate and plot the path they would take during a
# 45 degree shift.
segment_colors = cm.inferno(np.linspace(0, 1, n_steps))
for ori in oris:
    points = Orientation.stack([ori, (shift * ori)]).reduce()
    points.symmetry = Oh
    path = Orientation.from_path_ends(points, steps=n_steps)
    rod_ax.scatter(path, c=segment_colors)
    ipf_ax.scatter(path, c=segment_colors)

# add the wireframe and clean up the plot.
fz = OrientationRegion.from_symmetry(path.symmetry)
rod_ax.plot_wireframe(fz)
rod_ax._correct_aspect_ratio(fz)
rod_ax.axis("off")
rod_ax.set_title(r"Rodrigues, multiple paths")
ipf_ax.set_title(r"IPF, multiple paths        ")


# ============ #
# Example 2: Plotting a path using `Rotation.scatter'
# This subplot traces the path of an object rotated 90 degrees around the
# X axis, then 90 degrees around the Y axis.

rots = Orientation.from_axes_angles(
    [[1, 0, 0], [1, 0, 0], [0, 1, 0]], [0, 90, 90], degrees=True, symmetry=C1
)
rots[2] = rots[1] * rots[2]
path = Orientation.from_path_ends(rots, steps=n_steps)
# create a list of RGBA color values for a gradient red line and blue line
path_colors = np.vstack(
    [
        cm.Reds(np.linspace(0.5, 1, n_steps)),
        cm.Blues(np.linspace(0.5, 1, n_steps)),
    ]
)

# Here, we instead use the in-built plotting tool from
# Orientation.scatter to auto-generate the subplot. This is especially handy when
# plotting only a single Orientation object.
path.scatter(figure=fig, position=[2, 2, 3], marker=">", c=path_colors)
fig.axes[2].set_title(r"Axis-Angle, two $90^\circ$ rotations")


# ============ #
# Example 3: paths in stereographic plots

# This is similar to the second example, but now vectors are being rotated
# 30 degrees around the [1,1,1] axis on a stereographic plot.

vec_ax = plt.subplot(2, 2, 4, projection="stereographic", hemisphere="upper")
ipf_colormap = DirectionColorKeyTSL(C1)

# define a mesh of vectors with approximately 20 degree spacing, and
# within 80 degrees of the Z axis
vecs = sample_S2(20)
vecs = vecs[vecs.polar < (80 * np.pi / 180)]

# define a 15 degree rotation around [1,1,1]
rots = Quaternion.from_axes_angles([1, 1, 1], [0, 15], degrees=True)

for vec in vecs:
    path_ends = rots * vec
    # color each path using a gradient pased on the IPF coloring.
    c = ipf_colormap.direction2color(vec)
    if np.abs(path_ends.cross(path_ends[::-1])[0].norm) > 1e-12:
        path = Vector3d.from_path_ends(path_ends, steps=100)
        segment_c = c * np.linspace(0.25, 1, path.size)[:, np.newaxis]
        vec_ax.scatter(path, c=segment_c)
    else:
        vec_ax.scatter(path_ends[0], c=c)

vec_ax.set_title(r"Stereographic")
vec_ax.set_labels("X", "Y")
plt.tight_layout()
