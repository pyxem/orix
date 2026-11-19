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
=====================
Pole density function
=====================

The pole density function (PDF) quantifies the distribution of crystallographic
poles, for example in texture analysis. This example shows the poles from a
titanium sample as individual points and as density maps.
"""

import matplotlib.pyplot as plt
import numpy as np

from orix.crystal_map import Phase
from orix.data import ti_orientations
from orix.plot import register_projections
from orix.sampling import sample_S2
from orix.vector import Miller, Vector3d

register_projections()

# %%
# Load the titanium EBSD orientations with point group :math:`622`
# :cite:`krakow2017onthree`. The function :func:`~orix.data.ti_orientations`
# downloads the data to a local cache when needed.
#
# Generate the symmetry-equivalent :math:`\{01\bar{1}1\}` poles in the crystal
# frame. The outer product with the inverse orientations gives their directions
# in the sample frame. Dask splits the computation into chunks.

O = ti_orientations(allow_download=True)
g = Miller(hkil=(0, 1, -1, 1), phase=Phase(point_group=O.symmetry))
g = g.symmetrise(unique=True)
poles = O.inv().outer(g, lazy=True, chunk_size=2000, progressbar=False)

poles.scatter(
    hemisphere="both",
    s=4,
    alpha=0.02,
    figure_kwargs={"figsize": (10, 5)},
    axes_labels=["X", "Y"],
)

# %%
# Individual points overlap in the stereographic projection. The PDF uses
# equal-area bins on the unit sphere :math:`S_2` to measure the density
# :cite:`rohrer2004distribution`. Each bin has the same probability of receiving
# a uniformly distributed direction. The density can therefore be expressed in
# multiples of a random distribution (MRD). The equal-area sampling grid
# below uses a resolution of 10 degrees.

fig = plt.figure(figsize=(10, 5), layout="constrained")
ax0 = fig.add_subplot(121, projection="stereographic", hemisphere="upper")
ax1 = fig.add_subplot(122, projection="3d")
mesh = sample_S2(resolution=10, method="equal_area")

ax0.scatter(mesh, s=4)
ax0.set_labels("X", "Y", None)
ax0.show_hemisphere_label()

ax1.scatter(*mesh.data.T, s=4)
ax1.set(
    xlim=(-1, 1),
    ylim=(-1, 1),
    zlim=(-1, 1),
    xticks=(-1, 0, 1),
    yticks=(-1, 0, 1),
    zticks=(-1, 0, 1),
    xlabel="X",
    ylabel="Y",
    zlabel="Z",
)
ax1.set_box_aspect((1, 1, 1), zoom=0.85)

# %%
# With more uniformly distributed vectors, the estimated density approaches
# 1 MRD. All four maps use the same color limits.

rng = np.random.default_rng(0)
random_poles = Vector3d(rng.standard_normal((1_000_000, 3))).unit
fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 10),
    layout="constrained",
    subplot_kw={"projection": "stereographic"},
)
for ax, count in zip(axes.flat, (1_000, 10_000, 100_000, 1_000_000)):
    ax.pole_density_function(random_poles[:count], vmin=0.8, vmax=1.2)
    ax.set_labels("X", "Y", None)
    ax.set_title(f"{count:,} vectors")

# %%
# The angular bin size, Gaussian broadening, and colormap are controlled by
# ``resolution``, ``sigma``, and ``cmap``, respectively.

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 10),
    layout="constrained",
    subplot_kw={"projection": "stereographic"},
)
ax0, ax1, ax2, ax3 = axes.flat
ax0.pole_density_function(random_poles, resolution=1)
ax0.set_title("1° bins")
ax1.pole_density_function(random_poles, resolution=5)
ax1.set_title("5° bins")
ax2.pole_density_function(random_poles, resolution=1, sigma=15)
ax2.set_title(r"1° bins, $\sigma = 15^\circ$")
ax3.pole_density_function(random_poles, resolution=1, cmap="gray_r")
ax3.set_title("1° bins, gray_r colormap")
for ax in axes.flat:
    ax.set_labels("X", "Y", None)

# %%
# The titanium sample contains regions of similar crystal orientation, known
# as grains. The PDF of the measured poles shows concentrations in both
# hemispheres.

# sphinx_gallery_thumbnail_number = 5
poles.pole_density_function(
    hemisphere="both", figure_kwargs={"figsize": (10, 5), "layout": "constrained"}
)

# %%
# A logarithmic color scale compresses the contrast between regions of high and
# low density. The two upper-hemisphere views show the same poles.

fig, axes = plt.subplots(
    ncols=2,
    figsize=(10, 5),
    layout="constrained",
    subplot_kw={"projection": "stereographic", "hemisphere": "upper"},
)
axes[0].scatter(poles, s=2, alpha=0.02)
axes[0].set_title("Individual poles")
axes[1].pole_density_function(poles, log=True)
axes[1].set_title("Pole density")
for ax in axes:
    ax.set_labels("X", "Y", None)

# %%
# Overlay the individual points on the density map by passing the figure from
# :meth:`~orix.vector.Vector3d.scatter` to
# :meth:`~orix.vector.Vector3d.pole_density_function`.

fig = poles.scatter(
    s=4,
    alpha=0.01,
    c="white",
    return_figure=True,
    axes_labels=["X", "Y"],
    show_hemisphere_label=True,
    figure_kwargs={"figsize": (6, 5), "layout": "constrained"},
)
poles.pole_density_function(log=True, figure=fig)
