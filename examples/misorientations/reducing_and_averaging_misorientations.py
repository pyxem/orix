#
# Copyright 2018-2026 the orix developers
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

r"""
=======================================================
Reducing and averaging misorientations and orientations
=======================================================

This example introduces the concept of reducing an orientation or misorientation with
respect to symmetry, as well as the related concept of averaging a misorientation.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import orix.plot as opl
import orix.quaternion as oqu
import orix.quaternion.symmetry as osm

opl.register_projections()

# %%
# Orientation reduction
# =====================
# Orientation :meth:`~orix.quaternion.Orientation.reduce` and the equivalent for
# misorientations will rotate any transform with symmetry to the equivalent
# representation with the smallest possible angle of rotation.
# This is what is meant by "reducing" a (mis)orientation.

# Reproducible random data
np.random.seed(2319)

# Get a random (seeded) m-3m orientation and its symmetrically equivalent inside the 432
# fundamental zone
pg_m3m = osm.Oh
pg_432 = pg_m3m.proper_subgroup
ori_rand = oqu.Orientation.random(symmetry=pg_m3m)
ori_reduced = ori_rand.reduce()

print(ori_rand)
print(ori_reduced)

# Get the 432 fundamental zone
fz_432 = oqu.OrientationRegion.from_symmetry(end=pg_m3m)  # or pg_432

# Get symmetrically equivalent orientations, excluding itself
rot_all = oqu.Rotation(pg_432.outer(ori_rand))
rot_equiv = rot_all[~np.isclose(rot_all.dot(ori_reduced), 1)]

# Plot the orientation inside the 432 fundamental zone and equivalent rotations in
# homochoric axis-angle space
fig1, ax1 = plt.subplots(figsize=(6, 5), subplot_kw={"projection": "homochoric"})
ax1.axis("off")
ax1.set(aspect="equal", xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
ax1.plot_wireframe(fz_432)
ax1.scatter(ori_reduced, c="C1", label="Reduced")
ax1.scatter(rot_equiv, c="C0", label="Equivalent")
_ = ax1.legend(loc="upper center", ncols=2)

# %%
# The inclusion of symmetry combined with the periodic nature of rotations can make the
# definition of a mean or average ambiguous (more on this below), so the first step when
# an average is calculated is to reduce the transformations and then calculate the
# average.
# A side effect of this is the mean (mis)orientation returned by orix will also be a
# reduced representation.

# Get misorientations and their symmetrically equivalent inside the (3, 222) fundamental
# zone
mori_pgs = (osm.C3, osm.D2)  # (3, 222)
qu_data = np.stack([norm.rvs(i, 0.06, 20) for i in [0.39, 0.28, -0.39, 0.78]]).T
mori_clustered = oqu.Misorientation(qu_data, symmetry=mori_pgs)
mori_reduced = mori_clustered.reduce()
mori_mean, mori_neighbors = mori_clustered.mean(return_neighbors=True)

fz_3_222 = oqu.OrientationRegion.from_symmetry(*mori_pgs)

# Plot
fig2, ax2 = plt.subplots(figsize=(6, 5), subplot_kw={"projection": "homochoric"})
ax2.axis("off")
ax2.set(aspect="equal", xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
ax2.set_box_aspect((1, 1, 1), zoom=1.8)
ax2.plot_wireframe(fz_3_222)
ax2.scatter(oqu.Rotation(mori_clustered), c="C0", label="Initial")
ax2.scatter(oqu.Rotation(mori_reduced), c="C1", label="Reduced")
ax2.scatter(oqu.Rotation(mori_mean), c="C2", marker="X", s=100, label="Mean")
_ = ax2.legend(loc="upper center", ncols=3)

# %%
# The fundamental zone
# ====================
#
# In the plots above, a wireframe was included that defined a bounded volume within
# which all reduced representations fell.
# This is known as a fundamental zone (FZ), and contains a single unique (so-called
# fundamental) representation of every transformation with respect to a given symmetry
# or a combination of two symmetries.
# Representations that fall within a FZ are also guaranteed to have the smallest
# possible angle of rotation.
#
# There are several ways in which a fundamental zone can be defined, with most
# discrepencies stemming from how improper transforms (i.e., inversions and
# rotoinversions) should be handled.
# orix uses the rules presented in :cite:`krakow2017onthree`, but expanded to all 1024
# misorientation groups.
# This can be verified by comparing the following plots to Figure 5 of the same paper.

name2group = {x.name: x for x in osm._groups}

fig3 = plt.figure(figsize=(5, 8), layout="constrained")

base_pairs = [
    ["432", "3"],
    ["23", "3"],
    ["432", "1"],
    ["23", "1"],
    ["3", "422"],
    ["622", "1"],
    ["4", "211"],
    ["32", "1"],
    ["222", "1"],
    ["3", "4"],
    ["6", "1"],
    ["4", "1"],
    ["3", "1"],
    ["211", "1"],
    ["1", "1"],
]
for i, (name1, name2) in enumerate(base_pairs):
    pgI = name2group[name1]
    pgII = name2group[name2]
    fz_I_II = oqu.OrientationRegion.from_symmetry(pgI, pgII)

    ax = fig3.add_subplot(5, 3, i + 1, projection="homochoric")
    ax.axis("off")
    ax.plot_wireframe(fz_I_II)
    ax._correct_aspect_ratio(fz_I_II)  # NB! Private method, may become public in future
    ax.set_title(rf"{'abcdefghijklmno'[i]}: {name1} $\rightarrow$ {name2}")

# %%
# The same paper also contains a concise overview of the use of Rodrigues, Homochoric,
# or neo-Eulerian plots to visualize the placement and distribution of rotations in 3D
# space.
#
# It is not ran as part of this example since calculating and plotting 66 FZs is
# relatively computationally intensive, but the following figure matches Table 3 of the
# same paper, showing the subdivisions of the above 15 FZs caused by shared rotation
# elements between the point groups.
# We can plot it by uncommeting.

# names = ["432", "23", "622", "6", "32", "3", "422", "4", "222", "211", "1"]
# fig4 = plt.figure(figsize=(5 * 11, 5 * 11))
# for i in range(len(names)):
#     pgI = name2group[names[i]]
#     for j in range(len(names) - i):
#         pgII = name2group[names[-j - 1]]
#         fz_I_II = oqu.OrientationRegion.from_symmetry(pgI, pgII)
#         ax_j = fig4.add_subplot(11, 11, j * 11 + i + 1, projection="homochoric")
#         ax_j.axis("off")
#         # NB! Private method, may become public in future
#         ax_j._correct_aspect_ratio(fz_I_II)
#         ax_j.plot_wireframe(fz_I_II)

# %%
# Defining a mean in rotation space
# =================================
#
# Up until now, we have not defined what is meant by the mean of a group of transforms.
# Because rotation space is periodic, the concept of a Euclidean norm does not apply,
# and instead a Frobenius norm is used, which in this context can be thought of as the
# magnitude of the angular rotation necessary to align two transforms.
# Noteably, this is *not* simply the normalized average of two transform's quaternion
# representations, as is sometimes done in other software to get a fast approximation
# for clustered transforms.
# See the documentation of :meth:`~orix.quaternion.Quaternion.mean` for details.
#
# The extension of this is a mean transform defined as the transform whose total angular
# deviation from all transforms in the queried group is the minimum possible value.
#
# However, there is not a convenient algorithm to calculate this correctly, so instead
# orix does the following:
#
# 1. Transforms are reduced to the appropriate fundamental zone
# 2. A rough mean is calculated
# 3. Transforms with equivalents closer to the rough mean are updated to the closer
#    value
# 4. A precise mean is recalculated
#
# The plot below helps visualize this process.

pg_222 = osm.D2

np.random.seed(2319)
qu_data = np.stack([norm.rvs(i, 0.1, 20) for i in [0.1, 0.1, 0.2, 0.3]]).T
ori_cluster = oqu.Orientation(qu_data, symmetry=pg_222)
ori_reduced = ori_cluster.reduce()
ori_mean, ori_neighbors = ori_cluster.mean(return_neighbors=True)
ori_flipped = ori_neighbors[np.abs(ori_neighbors.angle - ori_reduced.angle) > 1e-3]

fig5, ax = plt.subplots(figsize=(5, 6), subplot_kw={"projection": "homochoric"})
fz_222 = oqu.OrientationRegion.from_symmetry(end=pg_222)
ax.axis("off")
ax.set(aspect="equal", xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
ax.plot_wireframe(fz_222)
ax.scatter(oqu.Rotation(ori_cluster), c="C0", label="Initial")
ax.scatter(oqu.Rotation(ori_reduced), c="C1", label="Reduced")
ax.scatter(oqu.Rotation(ori_flipped), c="C2", label="Flipped")
ax.scatter(oqu.Rotation(ori_mean), c="C3", marker="X", s=100, label="Mean")
_ = ax.legend(ncols=4)
