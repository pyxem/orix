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
=============================================
Clustering across fundamental zone boundaries
=============================================

This example shows how to perform density based clustering of crystal orientations with
and without the application of crystal symmetry using simulated data, as presented in
:cite:`johnstone2020density`.
"""

########################################################################################
# Import necessary dependencies and orix tools.

import matplotlib.animation as manimation
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from sklearn.cluster import DBSCAN

from orix.plot import register_projections
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import C1, Oh

register_projections()  # Register custom Matplotlib projections

########################################################################################
# Generate artificial data
# ========================
#
# Generate three random von Mises distributions of orientations as model clusters and
# set the *Oh* (:math:`m\bar{3}m``) point group symmetry.

n_orientations = 50
alpha = 65  # Lower value gives "looser" distribution

# Cluster 1
cluster1 = Rotation.random_vonmises(n_orientations, alpha=alpha)

# Cluster 2
centre2 = Rotation.from_axes_angles((1, 0, 0), np.pi / 4)
cluster2 = Rotation.random_vonmises(n_orientations, alpha=alpha, reference=centre2)

# Cluster 3
centre3 = Rotation.from_axes_angles((1, 1, 0), np.pi / 3)
cluster3 = Rotation.random_vonmises(n_orientations, alpha=alpha, reference=centre3)

# Stack and map into the Oh fundamental zone
ori = Orientation.stack([cluster1, cluster2, cluster3]).flatten()
ori.symmetry = Oh
ori = ori.reduce()

########################################################################################
# Orientation clustering
# ======================
#
# Without symmetry
# ----------------
#
# Compute misorientations, i.e. distance between orientations.

# Remove symmetry by setting it to point group 1 (identity operation)
ori_without_symmetry = Orientation(ori.data, symmetry=C1)

# Misorientations
mori1 = (~ori_without_symmetry).outer(ori_without_symmetry)

# Misorientation angles
D1 = mori1.angle

########################################################################################
# Perform clustering.

dbscan_naive = DBSCAN(eps=0.3, min_samples=10, metric="precomputed").fit(D1)
print("Labels:", np.unique(dbscan_naive.labels_))

########################################################################################
# With symmetry
# -------------
#
# Compute misorientations, i.e. distance between orientations, with symmetry.

mori2 = (~ori).outer(ori)

mori2.symmetry = Oh
mori2 = mori2.reduce()

D2 = mori2.angle

########################################################################################
# Perform clustering.

dbscan = DBSCAN(eps=np.deg2rad(17), min_samples=20, metric="precomputed").fit(
    D2.astype(np.float32)
)
print("Labels:", np.unique(dbscan.labels_))

########################################################################################
# This should have shown that without symmetry there are 6 clusters, whereas with
# symmetry there are 3.
#
# Visualisation
# =============
#
# Assign colours to each cluster.

color_names = [to_rgb(f"C{i}") for i in range(6)]  # ['C0', 'C1', ...]

colors_naive = label2rgb(dbscan_naive.labels_, colors=color_names, bg_label=-1)
colors = label2rgb(dbscan.labels_, colors=color_names, bg_label=-1)

########################################################################################
# Plot orientation clusters with Matplotlib and
# :meth:`~orix.quaternion.Misorientation.scatter`.

# Set symmetry to "trick" the scatter plot to use the Oh fundamental zone
ori_without_symmetry.symmetry = ori.symmetry

# Create figure with a height/width ratio of 1/2
fig = plt.figure(figsize=(12, 6))

# Add the fundamental zones with clusters to the existing figure
ori_without_symmetry.scatter(figure=fig, position=(1, 2, 1), c=colors_naive)
ori.scatter(figure=fig, position=122, c=colors)

########################################################################################
# Generate an animation of the plot (assuming an interactive Matplotlib backend is
# used).


def animate(angle):
    fig.axes[0].view_init(15, angle)
    fig.axes[1].view_init(15, angle)
    plt.draw()


ani = manimation.FuncAnimation(
    fig, animate, np.linspace(75, 360 + 74, 720), interval=25
)

plt.show()
