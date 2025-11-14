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
==========================
Clustering misorientations
==========================

This example shows how to cluster crystal misorientations.
The example data is from an orientation map of a highly deformed titanium sample, as
presented in :cite:`johnstone2020density`.
The data can be downloaded to your local cache using the :mod:`orix.data` module.

.. note::

    Clustering of (mis)orientations was what orix was initially written for.
"""

# %%
# Import necessary dependencies and orix tools.

from matplotlib.colors import to_rgb
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from sklearn.cluster import DBSCAN

from orix.data import ti_orientations
from orix.plot import IPFColorKeyTSL, register_projections
from orix.quaternion import Misorientation, Rotation
from orix.quaternion.symmetry import D6
from orix.vector import Vector3d

plt.rcParams.update({"font.size": 20, "figure.figsize": (10, 10)})

register_projections()  # Register custom Matplotlib projections

# %%
# Import data
# ===========
#
# Load Ti orientations with the point group symmetry *D6* (*622*).
# We have to explicitly allow downloading from an external source.

ori = ti_orientations(allow_download=True)
print(ori)

# %%
# The orientations define transformations from the sample to the Ti crystal reference
# frame, sample -> crystal.
# The above referenced paper assumes the opposite convention, crystal -> sample (which
# is the one used in e.g. MTEX).
# So, we have to invert the orientations.

ori = ~ori

# %%
# Then, reshape the orientation to the correct spatial dimension for the scan.

ori = ori.reshape(381, 507)

# %%
# And select a subset of the orientations with a suitable size for this demonstration.

ori = ori[-100:, :200]

# %%
# Get an overview of the orientations from inverse pole figure (IPF) maps.

ckey = IPFColorKeyTSL(D6)

directions = [(1, 0, 0), (0, 1, 0)]
titles = ["X", "Y"]

fig, axes = plt.subplots(ncols=2, figsize=(15, 4.5), layout="constrained")
for i, ax in enumerate(axes):
    ckey.direction = Vector3d(directions[i])
    # Invert because orix assumes sample -> crystal when coloring orientations
    ax.imshow(ckey.orientation2color(~ori))
    ax.set_title(f"IPF-{titles[i]}")
    ax.axis("off")

# Add color key
ax_ipfkey = fig.add_axes(
    [0.85, 0.2, 0.13, 0.13],  # (Left, bottom, width, height)
    projection="ipf",
    symmetry=ori.symmetry.laue,
)
ax_ipfkey.plot_ipf_color_key(show_title=False)

# %%
# Map the orientations into the fundamental zone (find symmetrically equivalent
# orientations with the smallest angle of rotation) of *D6*.

ori = ori.reduce()

# %%
# Compute misorientations (in the horizontal direction).

mori_all = Misorientation(~ori[:, :-1] * ori[:, 1:])

# %%
# Keep only misorientations with a disorientation angle higher than :math:`7^{\circ}``,
# assumed to represent grain boundaries.

boundary_mask = mori_all.angle > np.deg2rad(7)
mori = mori_all[boundary_mask]

# %%
# Map the misorientations into the fundamental zone of (*D6*, *D6*).

mori.symmetry = (D6, D6)
mori = mori.reduce()

# %%
# Compute distance matrix
# =======================
#
# Increase the chunk size for a faster but more memory intensive computation.

D = mori.get_distance_matrix()

# %%
# Clustering
# ==========
#
# Apply mask to remove small misorientations associated with grain orientation spread.

small_mask = mori.angle < np.deg2rad(7)
D = D[~small_mask][:, ~small_mask]
mori = mori[~small_mask]

# %%
# For parameter explanations of the DBSCAN algorithm (Density-Based Spatial Clustering
# for Applications with Noise), see the :class:`sklearn.cluster.DBSCAN` documentation.

# Compute clusters
dbscan = DBSCAN(eps=0.05, min_samples=10, metric="precomputed").fit(D)

unique_labels, all_cluster_sizes = np.unique(dbscan.labels_, return_counts=True)
print("Labels:", unique_labels)

n_clusters = unique_labels.size - 1
print("Number of clusters:", n_clusters)

# %%
# Calculate the mean misorientation associated with each cluster.

unique_cluster_labels = unique_labels[1:]  # Without the "no-cluster" label -1
cluster_sizes = all_cluster_sizes[1:]

rc = Rotation.from_axes_angles((0, 0, 1), 15, degrees=True)

mori_mean = []
for label in unique_cluster_labels:
    # Rotate
    mori_i = rc * mori[dbscan.labels_ == label]

    # Map into the fundamental zone
    mori_i.symmetry = (D6, D6)
    mori_i = mori_i.reduce()

    # Get the cluster mean
    mori_i = mori_i.mean()

    # Rotate back and add to list
    cluster_mean_local = (~rc) * mori_i
    mori_mean.append(cluster_mean_local)

cluster_means = Misorientation.stack(mori_mean).flatten()

# Map into the fundamental zone
cluster_means.symmetry = (D6, D6)
cluster_means = cluster_means.reduce()
print(cluster_means)

# %%
# Inspect misorientations in the axis-angle representation.

print(cluster_means.axis)

print(np.rad2deg(cluster_means.angle))

# %%
# Define reference misorientations associated with twinning orientation relationships.

# From Krakow et al.
twin_theory = Rotation.from_axes_angles(
    axes=[
        (1, 0, 0),  # sigma7a
        (1, 0, 0),  # sigma11a
        (2, 1, 0),  # sigma11b
        (1, 0, 0),  # sigma13a
        (2, 1, 0),  # sigma13b
    ],
    angles=[64.40, 34.96, 85.03, 76.89, 57.22],
    degrees=True,
)

# %%
# Calculate difference, defined as minimum rotation angle, between measured and
# theoretical values.

mori2 = (~twin_theory).outer(cluster_means)
sym_op = D6.outer(D6).unique()
mori2_equiv = D6.outer(~twin_theory).outer(sym_op).outer(cluster_means).outer(D6)
D2 = mori2_equiv.angle.min(axis=(0, 2, 4))

print(np.rad2deg(D2))

# %%
# We see that the first, second, third, and fourth clusters are within
# :math:`4.5^{\circ}` of :math:`\Sigma7`a, :math:`\Sigma13`a,
# :math:`\Sigma11`a, and :math:`\Sigma13`b, respectively.
#
# Visualisation
# =============
#
# Associate colours with clusters for plotting.

colors = [to_rgb(f"C{i}") for i in range(cluster_means.size)]
labels_rgb = label2rgb(dbscan.labels_, colors=colors, bg_label=-1)

# %%
# Inspect misorientation axes of clusters in an inverse pole figure.

cluster_sizes = all_cluster_sizes[1:]
cluster_sizes_scaled = 1000 * cluster_sizes / cluster_sizes.max()

fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "ipf", "symmetry": D6})
ax.scatter(cluster_means.axis, c=colors, s=cluster_sizes_scaled, alpha=0.5, ec="k")

# %%
# Plot a top view of the misorientation clusters within the fundamental zone for the
# (*D6*, *D6*) bicrystal symmetry.

wireframe_kws = {"color": "k", "lw": 0.5, "alpha": 0.1, "rcount": 361, "ccount": 361}
fig = mori.scatter(
    projection="axangle",
    wireframe_kwargs=wireframe_kws,
    c=labels_rgb.reshape(-1, 3),
    s=4,
    alpha=0.5,
    return_figure=True,
)
ax = fig.axes[0]
ax.view_init(elev=90, azim=-60)

handle_kwds = {"marker": "o", "color": "none", "markersize": 10}
handles = []
for i in range(n_clusters):
    line = mlines.Line2D(
        [0], [0], label=i + 1, markerfacecolor=colors[i], **handle_kwds
    )
    handles.append(line)
_ = ax.legend(
    handles=handles,
    loc="upper left",
    numpoints=1,
    labelspacing=0.15,
    columnspacing=0.15,
    handletextpad=0.05,
)
