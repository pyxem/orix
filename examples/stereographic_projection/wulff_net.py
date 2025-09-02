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
=========
Wulff net
=========

This example shows how to draw a Wulff net in the stereographic projection with great
and small circles.
"""

import matplotlib.pyplot as plt
import numpy as np

from orix import plot
from orix.vector import Vector3d

n = int(90 / 2)  # Degree / net resolution
steps = 500
kwargs = dict(linewidth=0.25, color="k")

polar = np.linspace(0, 0.5 * np.pi, num=n)
v_right = Vector3d.from_polar(azimuth=np.zeros(n), polar=polar)
v_left = Vector3d.from_polar(azimuth=np.ones(n) * np.pi, polar=polar)
v010 = Vector3d.zero(shape=(n,))
v010.y = 1
v010_opposite = -v010

fig, ax = plt.subplots(
    figsize=(5, 5), subplot_kw=dict(projection="stereographic"), layout="tight"
)
ax.stereographic_grid(False)
ax.draw_circle(v_right, steps=steps, **kwargs)
ax.draw_circle(v_left, steps=steps, **kwargs)
ax.draw_circle(v010, opening_angle=polar, steps=steps, **kwargs)
ax.draw_circle(v010_opposite, opening_angle=polar, steps=steps, **kwargs)
for label, azimuth, va, ha, offset in zip(
    ["B", "M''", "A", "M'"],
    np.array([0, 0.5, 1, 1.5]) * np.pi,
    ["center", "bottom", "center", "top"],
    ["left", "center", "right", "center"],
    [(0.02, 0), (0, 0.02), (-0.02, 0), (0, -0.02)],
):
    ax.text(azimuth, 0.5 * np.pi, s=label, offset=offset, c="r", va=va, ha=ha)
