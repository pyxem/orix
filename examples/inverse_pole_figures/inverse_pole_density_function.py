"""
=============================
Inverse pole density function
=============================

This example shows how to plot an inverse pole density function (IPDF)
:cite:`rohrer2004distribution` to inspect the distribution of crystal directions
pointing in some sample direction.
"""

import matplotlib.pyplot as plt

from orix import data, plot
from orix.vector import Vector3d

xmap = data.sdss_ferrite_austenite(allow_download=True)
print(xmap)

# Extract orientations
pg_laue = xmap.phases[1].point_group.laue
ori_fe = xmap["ferrite"].orientations
ori_au = xmap["austenite"].orientations

# Select sample direction
vec_sample = Vector3d([0, 0, 1])
vec_title = "Z"

# Rotate sample direction into every crystal
vec_crystal_fe = ori_fe * vec_sample
vec_crystal_au = ori_au * vec_sample

# Set IPDF range
vmin, vmax = (0, 3)

subplot_kw = dict(projection="ipf", symmetry=pg_laue)
fig = plt.figure(figsize=(9, 8))

ax0 = fig.add_subplot(221, direction=vec_sample, **subplot_kw)
ax0.scatter(ori_fe, alpha=0.05)
_ = ax0.set_title(f"Ferrite, {vec_title}")

ax1 = fig.add_subplot(222, direction=vec_sample, **subplot_kw)
ax1.scatter(ori_au, alpha=0.05)
_ = ax1.set_title(f"Austenite, {vec_title}")

ax2 = fig.add_subplot(223, direction=vec_sample, **subplot_kw)
ax2.pole_density_function(vec_crystal_fe, vmin=vmin, vmax=vmax)
_ = ax2.set_title(f"Ferrite, {vec_title}")

ax3 = fig.add_subplot(224, direction=vec_sample, **subplot_kw)
ax3.pole_density_function(vec_crystal_au, vmin=vmin, vmax=vmax)
_ = ax3.set_title(f"Austenite, {vec_title}")
