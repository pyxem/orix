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
r"""
========================
Plot symmetry operations
========================

This example shows how to draw proper symmetry operations :math:`s`
(no reflections or inversions).
"""

import matplotlib.pyplot as plt

from orix.plot import register_projections
from orix.vector import Vector3d

register_projections()  # Register our custom Matplotlib projections

marker_size = 200
fig, (ax0, ax1) = plt.subplots(
    ncols=2,
    subplot_kw={"projection": "stereographic"},
    layout="tight",
)

ax0.set_title("432", pad=20)
# 4-fold (outer markers will be clipped a bit...)
v4fold = Vector3d([[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
ax0.symmetry_marker(v4fold, fold=4, c="C4", s=marker_size)
ax0.draw_circle(v4fold, color="C4")
# 3-fold
v3fold = Vector3d([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]])
ax0.symmetry_marker(v3fold, fold=3, c="C3", s=marker_size)
ax0.draw_circle(v3fold, color="C3")
# 2-fold
# fmt: off
v2fold = Vector3d(
    [
        [ 1,  0, 1],
        [ 0,  1, 1],
        [-1,  0, 1],
        [ 0, -1, 1],
        [ 1,  1, 0],
        [-1, -1, 0],
        [-1,  1, 0],
        [ 1, -1, 0],
    ]
)
# fmt: on
ax0.symmetry_marker(v2fold, fold=2, c="C2", s=marker_size)
ax0.draw_circle(v2fold, color="C2")

ax1.set_title("222", pad=20)
# 2-fold
v2fold = Vector3d([[0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
ax1.symmetry_marker(v2fold, fold=2, c="C2", s=2 * marker_size)
ax1.draw_circle(v2fold, color="C2")
