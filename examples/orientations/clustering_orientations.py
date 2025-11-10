#
# Copyright 2018-2025 the orix developers
#
# This file is part of orix.
#
# orix is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# orix is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix. If not, see <http://www.gnu.org/licenses/>.
#

"""
=======================
Clustering orientations
=======================

This example shows hot to cluster crystal orientations.
The example data is from an orientation map of a highly deformed titanium sample, as
presented in :cite:`johnstone2020density`.
The data can be downloaded to your local cache using the :mod:`orix.data` module.
"""

########################################################################################
# Import necessary dependencies and orix tools.

from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from skimage.color import label2rgb
from sklearn.cluster import DBSCAN

from orix import data

# Import orix classes
from orix.plot import IPFColorKeyTSL, register_projections
from orix.quaternion import Orientation
from orix.quaternion.symmetry import D6
from orix.vector import AxAngle, Vector3d

plt.rcParams.update(
    {"font.size": 20, "figure.figsize": (10, 10), "figure.facecolor": "w"}
)
register_projections()  # Register custom Matplotlib projections

########################################################################################
# Import data
# ===========
#
# Load Ti orientations with the point group symmetry *D6* (*622*).
# We have to explicitly allow downloading from an external source.

ori = data.ti_orientations(allow_download=True)
print(ori)

########################################################################################
# The orientations define transformations from the sample to the crystal reference
# frame, sample -> crystal.
# The above referenced paper assumes the opposite convention, crystal -> sample (which
# is the one used in e.g. MTEX).
# So, we have to invert the orientations.

ori = ~ori

########################################################################################
# Then, reshape the orientations to the correct spatial dimension for the scan.

ori = ori.reshape(381, 507)

########################################################################################
# And select a subset of the orientations to a suitable size for this example.

ori = ori[-100:, :200]

########################################################################################
# Get an overview of the orientations from inverse pole figure (IPF) maps.

ckey = IPFColorKeyTSL(D6)

directions = [(1, 0, 0), (0, 1, 0)]
titles = ["X", "Y"]

fig, axes = plt.subplots(ncols=2, figsize=(15, 10), layout="constrained")
for i, ax in enumerate(axes):
    ckey.direction = Vector3d(directions[i])
    # Invert because orix assumes sample -> crystal when coloring orientations
    ax.imshow(ckey.orientation2color(~ori))
    ax.set_title(f"IPF-{titles[i]}")
    ax.axis("off")

# Add color key
ax_ipfkey = fig.add_axes(
    [1.04, 0.34, 0.1, 0.1],  # (Left, bottom, width, height)
    projection="ipf",
    symmetry=ori.symmetry.laue,
)
ax_ipfkey.plot_ipf_color_key(show_title=False)

########################################################################################
# Map the orientations into the fundamental zone (find symmetrically equivalent
# orientations with the smallest angle of rotation) of *D6*

ori = ori.reduce()

########################################################################################
# Compute distance matrix
# =======================

# Increase the chunk size for a faster but more memory intensive computation
D = ori.get_distance_matrix(lazy=True, chunk_size=20)

D = D.reshape(ori.size, ori.size)

########################################################################################
# Clustering
# ==========
#
# For parameter explanations of the DBSCAN algorithm (Density-Based Spatial Clustering
# for Applications with Noise), see the :class:`sklearn.cluster.DBSCAN` documentation.

# This call will use about 6 GB of memory, but the data precision of
# the D matrix can be reduced from float64 to float32 save memory:
D = D.astype(np.float32)

dbscan = DBSCAN(
    eps=0.05,  # Max. distance between two samples in radians
    min_samples=40,
    metric="precomputed",
).fit(D)

unique_labels, all_cluster_sizes = np.unique(dbscan.labels_, return_counts=True)
print("Labels:", unique_labels)

all_labels = dbscan.labels_.reshape(ori.shape)
n_clusters = unique_labels.size - 1
print("Number of clusters:", n_clusters)

########################################################################################
# Calculate the mean orientation of each cluster.

unique_cluster_labels = unique_labels[1:]  # Without the "no-cluster" label -1
cluster_sizes = all_cluster_sizes[1:]

q_mean = [ori[all_labels == l].mean() for l in unique_cluster_labels]
cluster_means = Orientation.stack(q_mean).flatten()

# Map into the fundamental zone
cluster_means.symmetry = D6
cluster_means = cluster_means.reduce()
print(cluster_means)

########################################################################################
# Inspect rotation axes in the axis-angle representation.

print(cluster_means.axis)

########################################################################################
# Recenter data relative to the matrix cluster and recompute means.

ori_recentered = (~cluster_means[0]) * ori

# Map into the fundamental zone
ori_recentered.symmetry = D6
ori_recentered = ori_recentered.reduce()

cluster_means_recentered = Orientation.stack(
    [ori_recentered[all_labels == l].mean() for l in unique_cluster_labels]
).flatten()
print(cluster_means_recentered)

########################################################################################
# Inspect recentered rotation axes in the axis-angle representation.

cluster_means_recentered_axangle = AxAngle.from_rotation(cluster_means_recentered)
print(cluster_means_recentered_axangle.axis)

########################################################################################
# Visualisation
# =============
#
# Specify colours and lines to identify each cluster.

colors = []
lines = []
for i, cm in enumerate(cluster_means_recentered_axangle):
    colors.append(to_rgb(f"C{i}"))
    lines.append([(0, 0, 0), tuple(cm.data[0])])
labels_rgb = label2rgb(all_labels, colors=colors, bg_label=-1)

########################################################################################
# Inspect rotation axes of clusters (in the axis-angle representation) in an inverse
# pole figure.

cluster_sizes_scaled = 5000 * cluster_sizes / cluster_sizes.max()
fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection="ipf", symmetry=D6))
_ = ax.scatter(cluster_means.axis, c=colors, s=cluster_sizes_scaled, alpha=0.5, ec="k")

########################################################################################
# Plot a top view of the recentered orientation clusters within the fundamental zone for
# the *D6* (*622*) point group symmetry of Ti.
# The mean orientation of the largest parent grain is taken as the reference
# orientation.

wireframe_kwargs = dict(color="black", linewidth=0.5, alpha=0.1, rcount=181, ccount=361)
fig = ori_recentered.scatter(
    projection="axangle",
    wireframe_kwargs=wireframe_kwargs,
    c=labels_rgb.reshape(-1, 3),
    s=1,
    return_figure=True,
)
ax = fig.axes[0]
ax.view_init(elev=90, azim=-30)
ax.add_collection3d(Line3DCollection(lines, colors=colors))

handle_kwds = dict(marker="o", color="none", markersize=10)
handles = []
for i in range(n_clusters):
    line = Line2D([0], [0], label=i + 1, markerfacecolor=colors[i], **handle_kwds)
    handles.append(line)
_ = ax.legend(
    handles=handles,
    loc="lower right",
    ncol=2,
    numpoints=1,
    labelspacing=0.15,
    columnspacing=0.15,
    handletextpad=0.05,
)

########################################################################################
# Plot side view of orientation clusters.

fig2 = ori_recentered.scatter(
    return_figure=True,
    wireframe_kwargs=wireframe_kwargs,
    c=labels_rgb.reshape(-1, 3),
    s=1,
)
ax2 = fig2.axes[0]
ax2.add_collection3d(Line3DCollection(lines, colors=colors))
ax2.view_init(elev=0, azim=-30)

########################################################################################
# Plot map indicating spatial locations associated with each cluster.

fig3, ax3 = plt.subplots(figsize=(15, 10))
ax3.imshow(labels_rgb)
_ = ax3.axis("off")

plt.show()
