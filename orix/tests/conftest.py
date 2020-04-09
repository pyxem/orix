# -*- coding: utf-8 -*-
# Copyright 2018-2020 The pyXem developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pytest


@pytest.fixture(
    params=[
        """4.485496 0.952426 0.791507     0.000     0.000   22.2  0.060  1       6
1.343904 0.276111 0.825890    19.000     0.000   16.3  0.020  1       2""",
    ]
)
def angfile_astar(tmpdir, request):
    f = tmpdir.mkdir("angfiles").join("angfile_astar.ang")
    f.write(
        """# File created from ACOM RES results
# ni-dislocations.res
#
#
# MaterialName      Nickel
# Formula
# Symmetry          43
# LatticeConstants  3.520  3.520  3.520  90.000  90.000  90.000
# NumberFamilies    4
# hklFamilies       1  1  1 1 0.000000
# hklFamilies       2  0  0 1 0.000000
# hklFamilies       2  2  0 1 0.000000
# hklFamilies       3  1  1 1 0.000000
#
# GRID: SqrGrid#"""
    )
    f.write(request.param)
    return str(f)


@pytest.fixture(
    params=[
        # Tuple with four parameters: map_shape, step_sizes, phase_id, n_unknown_columns
        (
            (10, 10),  # map_shape
            (0.1, 0.1),  # step_sizes
            np.zeros(10 * 10, dtype=int),  # phase_id
            5,  # Number of unknown columns (one between ci and fit, the rest after fit)
        )
    ]
)
def angfile_tsl(tmpdir, request):
    """Create a dummy EDAX TSL .ang file from input.

    Parameters expected in 'request'
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    step_sizes : tuple of floats
        Step sizes in x and y coordinates.
    phase_id : np.ndarray
        Array of map size with phase IDs in header.
    n_unknown_columns : int
        Number of columns with unknown values.

    """
    f = tmpdir.mkdir("angfiles").join("angfile_tsl.ang")

    # Unpack parameters
    (ny, nx), (dy, dx), phase_id, n_unknown_columns = request.param

    # File header
    # TODO: Exchange for a multiphase header
    header = (
        "# TEM_PIXperUM          1.000000"
        "# x-star                0.413900"
        "# y-star                0.729100"
        "# z-star                0.514900"
        "# WorkingDistance       27.100000"
        "#"
        "# Phase 1"
        "# MaterialName      Aluminum"
        "# Formula       Al"
        "# Info      "
        "# Symmetry              43"
        "# LatticeConstants      4.040 4.040 4.040  90.000  90.000  90.000"
        "# NumberFamilies        69"
        "# hklFamilies        1 -1 -1 1 8.469246 1"
        "# ElasticConstants  -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000"
        "# Categories0 0 0 0 0 "
        "#"
        "# GRID: SqrGrid"
        "# XSTEP: 0.100000"
        "# YSTEP: 0.100000"
        "# NCOLS_ODD: 997"
        "# NCOLS_EVEN: 997"
        "# NROWS: 2"
        "#"
        "# OPERATOR:     sem"
        "#"
        "# SAMPLEID:     "
        "#"
        "# SCANID:   "
        "#"
    )

    # File columns
    map_size = ny * nx
    x = np.tile(np.arange(nx) * dx, ny)
    y = np.sort(np.tile(np.arange(ny) * dy, nx))
    e13 = np.tile(np.linspace(0, 2 * np.pi, nx), ny)
    e2 = np.tile(np.linspace(0, np.pi, nx), ny)
    ci = np.random.random(map_size)
    iq = np.random.uniform(1e3, 1e6, map_size)
    un = np.zeros(map_size, dtype=int)
    fit = np.random.uniform(0, 3, map_size)

    # Insert 10% non-indexed points
    non_indexed_points = np.random.choice(
        np.arange(map_size), replace=False, size=int(map_size * 0.1)
    )
    e13[non_indexed_points] = 4 * np.pi
    e2[non_indexed_points] = 4 * np.pi
    ci[non_indexed_points] = -1
    fit[non_indexed_points] = 180.0

    np.savetxt(
        f,
        X=np.column_stack(
            (e13, e2, e13, x, y, iq, ci, phase_id, un, fit)
            + (un,) * (n_unknown_columns - 1)
        ),
        fmt=(
            "%9.5f%10.5f%10.5f%13.5f%13.5f%9.1f%7.3f%3i%7i%8.3f"
            + "%10.5f" * (n_unknown_columns - 1)
            + " "
        ),
        header=header,
        comments="",
    )

    return str(f)
