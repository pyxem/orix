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
Plot crystal symmetry
=====================

This example demonstrates some advanced methods for altering
stereographic plots of symmetries. For a more general example, see
:ref:`sphx_glr_plot_symmetry_operations.py`. For examples of how to make
symmetry objects, see :ref:`create_symmetry.py`
"""
import matplotlib.pyplot as plt
import orix.quaternion as oqu
import orix.vector as ove
import orix.plot as opl

opl.register_projections()  # Register our custom Matplotlib projections

########################################################################################
# TODO: finish this example
pg_Oh = oqu.symmetry.PointGroups.get("m-3m")
v = ove.Vector3d.random(10)
v_symm = pg_Oh.outer(v).flatten()
fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "stereographic"})
pg_Oh.plot(ax=ax, show_name=False)
ax.set_title("my cool custom title")
ax.scatter(v_symm)
