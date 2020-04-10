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

import os
import tempfile

import numpy as np
import pytest


# TODO: Exchange for a multiphase header (change `phase_id` accordingly)
ANGFILE_TSL_HEADER = (
    "# TEM_PIXperUM          1.000000\n"
    "# x-star                0.413900\n"
    "# y-star                0.729100\n"
    "# z-star                0.514900\n"
    "# WorkingDistance       27.100000\n"
    "#\n"
    "# Phase 1\n"
    "# MaterialName      Aluminum\n"
    "# Formula       Al\n"
    "# Info      \n"
    "# Symmetry              43\n"
    "# LatticeConstants      4.040 4.040 4.040  90.000  90.000  90.000\n"
    "# NumberFamilies        69\n"
    "# hklFamilies        1 -1 -1 1 8.469246 1\n"
    "# ElasticConstants  -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000\n"
    "# Categories0 0 0 0 0 \n"
    "#\n"
    "# GRID: SqrGrid\n"
    "# XSTEP: 0.100000\n"
    "# YSTEP: 0.100000\n"
    "# NCOLS_ODD: 42\n"
    "# NCOLS_EVEN: 42\n"
    "# NROWS: 13\n"
    "#\n"
    "# OPERATOR:     sem\n"
    "#\n"
    "# SAMPLEID:     \n"
    "#\n"
    "# SCANID:   \n"
    "#\n"
)

ANGFILE_ASTAR_HEADER = (
    "# File created from ACOM RES results\n"
    "# ni-dislocations.res\n"
    "#     \n"
    "#     \n"
    "# MaterialName      Nickel\n"
    "# Formula\n"
    "# Symmetry          43\n"
    "# LatticeConstants  3.520  3.520  3.520  90.000  90.000  90.000\n"
    "# NumberFamilies    4\n"
    "# hklFamilies       1  1  1 1 0.000000\n"
    "# hklFamilies       2  0  0 1 0.000000\n"
    "# hklFamilies       2  2  0 1 0.000000\n"
    "# hklFamilies       3  1  1 1 0.000000\n"
    "#\n"
    "# GRID: SqrGrid#\n"
)

ANGFILE_EMSOFT_HEADER = (
    "# TEM_PIXperUM          1.000000\n"
    "# x-star                 0.446667\n"
    "# y-star                 0.586875\n"
    "# z-star                 0.713450\n"
    "# WorkingDistance        0.000000\n"
    "#\n"
    "# Phase 1\n"
    "# MaterialName    austenite\n"
    "# Formula       austenite\n"
    "# Info          patterns indexed using EMsoft::EMEBSDDI\n"
    "# Symmetry              43\n"
    "# LatticeConstants      3.595 3.595 3.595   90.000 90.000 90.000\n"
    "# NumberFamilies        0\n"
    "# Phase 2\n"
    "# MaterialName    ferrite/ferrite\n"
    "# Formula       ferrite/ferrite\n"
    "# Info          patterns indexed using EMsoft::EMEBSDDI\n"
    "# Symmetry              43\n"
    "# LatticeConstants      2.867 2.867 2.867   90.000 90.000 90.000\n"
    "# NumberFamilies        0\n"
    "# GRID: SqrGrid\n"
    "# XSTEP:  1.500000\n"
    "# YSTEP:  1.500000\n"
    "# NCOLS_ODD:   13\n"
    "# NCOLS_EVEN:   13\n"
    "# NROWS:   42\n"
    "#\n"
    "# OPERATOR:   Håkon Wiik Ånes\n"
    "#\n"
    "# SAMPLEID:\n"
    "#\n"
    "# SCANID:\n"
    "#\n"
)


@pytest.fixture
def temp_ang_file():
    with tempfile.TemporaryDirectory() as tempdir:
        fname = os.path.join(tempdir, "angfile.ang")
        f = open(fname, mode="w+")
        yield f


@pytest.fixture(
    params=[
        # Tuple with default values for five parameters: map_shape, step_sizes,
        # phase_id, n_unknown_columns, and example_rotations (see docstring below)
        (
            (13, 42),  # map_shape
            (0.1, 0.1),  # step_sizes
            np.random.choice([0], 13 * 42),  # phase_id
            5,  # Number of unknown columns (one between ci and fit, the rest after fit)
            np.array(
                [[4.48549, 0.95242, 0.79150], [1.34390, 0.27611, 0.82589],]
            ),  # example_rotations as rows of Euler angle triplets
        )
    ]
)
def angfile_tsl(tmpdir, request):
    """Create a dummy EDAX TSL .ang file from input.

    10% of map points are set to non-indexed (confidence index equal to
    -1).

    Parameters expected in `request`
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    step_sizes : tuple of floats
        Step sizes in x and y coordinates in microns.
    phase_id : np.ndarray
        Array of map size with phase IDs in header.
    n_unknown_columns : int
        Number of columns with unknown values.
    example_rotations : np.ndarray
        A sample, smaller than the map size, of example rotations as
        rows of Euler angle triplets.

    """
    f = tmpdir.mkdir("angfiles").join("angfile_tsl.ang")

    # Unpack parameters
    (ny, nx), (dy, dx), phase_id, n_unknown_columns, example_rotations = request.param

    # File columns
    map_size = ny * nx
    x = np.tile(np.arange(nx) * dx, ny)
    y = np.sort(np.tile(np.arange(ny) * dy, nx))
    ci = np.random.random(map_size)  # [0, 1]
    iq = np.random.uniform(low=1e3, high=1e6, size=map_size)
    un = np.zeros(map_size, dtype=int)
    fit = np.random.uniform(low=0, high=3, size=map_size)
    # Rotations
    rot_idx = np.random.choice(np.arange(len(example_rotations)), map_size)
    rot = example_rotations[rot_idx]

    # Insert 10% non-indexed points
    non_indexed_points = np.random.choice(
        np.arange(map_size), replace=False, size=int(map_size * 0.1)
    )
    rot[non_indexed_points] = 4 * np.pi
    ci[non_indexed_points] = -1
    fit[non_indexed_points] = 180.0

    np.savetxt(
        fname=f,
        X=np.column_stack(
            (rot[:, 0], rot[:, 1], rot[:, 2], x, y, iq, ci, phase_id, un, fit)
            + (un,) * (n_unknown_columns - 1)
        ),
        fmt=(
            "%9.5f%10.5f%10.5f%13.5f%13.5f%9.1f%7.3f%3i%7i%8.3f"
            + "%10.5f" * (n_unknown_columns - 1)
            + " "
        ),
        header=ANGFILE_TSL_HEADER,
        comments="",
    )

    return f


@pytest.fixture(
    params=[
        # Tuple with default values for four parameters: map_shape, step_sizes,
        # phase_id, and example_rotations (see docstring below)
        (
            (10, 10),  # map_shape
            (2.86, 2.86),  # step_sizes
            np.ones(10 * 10, dtype=int),  # phase_id
            np.array(
                [[6.148271, 0.792205, 1.324879], [6.155951, 0.793078, 1.325229],]
            ),  # example_rotations as rows of Euler angle triplets
        )
    ]
)
def angfile_astar(tmpdir, request):
    """Create a dummy NanoMegas ASTAR Index .ang file from input.

    Parameters expected in `request`
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    step_sizes : tuple of floats
        Step sizes in x and y coordinates in nanometres.
    phase_id : np.ndarray
        Array of map size with phase IDs in header.
    example_rotations : np.ndarray
        A sample, smaller than the map size, of example rotations as
        rows of Euler angle triplets.

    """
    f = tmpdir.mkdir("angfiles").join("angfile_astar.ang")

    # Unpack parameters
    (ny, nx), (dy, dx), phase_id, example_rotations = request.param

    # File columns
    map_size = ny * nx
    x = np.tile(np.arange(nx) * dx, ny)
    y = np.sort(np.tile(np.arange(ny) * dy, nx))
    ind = np.random.uniform(low=0, high=100, size=map_size)
    rel = np.round(np.random.random(map_size), decimals=2)  # [0, 1]
    relx100 = (rel * 100).astype(int)

    # Rotations
    n_rotations = len(example_rotations)
    if n_rotations == map_size:
        rot = example_rotations
    else:
        # Sample as many rotations from `example_rotations` as `map_size`
        rot_idx = np.random.choice(np.arange(len(example_rotations)), map_size)
        rot = example_rotations[rot_idx]

    np.savetxt(
        fname=f,
        X=np.column_stack(
            (rot[:, 0], rot[:, 1], rot[:, 2], x, y, ind, rel, phase_id, relx100)
        ),
        fmt="%8.6f%9.6f%9.6f%10.3f%10.3f%7.1f%7.3f%3i%8i",
        header=ANGFILE_ASTAR_HEADER,
        comments="",
    )

    return f


@pytest.fixture(
    params=[
        # Tuple with default values for four parameters: map_shape, step_sizes,
        # phase_id, and example_rotations (see docstring below)
        (
            (42, 13),  # map_shape
            (1.5, 1.5),  # step_sizes
            np.random.choice([1, 2], 42 * 13),  # phase_id
            np.array(
                [[6.148271, 0.792205, 1.324879], [6.155951, 0.793078, 1.325229],]
            ),  # example_rotations as rows of Euler angle triplets
        )
    ]
)
def angfile_emsoft(tmpdir, request):
    """Create a dummy EMsoft .ang file from input.

    Parameters expected in `request`
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    step_sizes : tuple of floats
        Step sizes in x and y coordinates in nanometres.
    phase_id : np.ndarray
        Array of map size with phase IDs in header.
    example_rotations : np.ndarray
        A sample, smaller than the map size, of example rotations as
        rows of Euler angle triplets.

    """
    f = tmpdir.mkdir("angfiles").join("angfile_emsoft.ang")

    # Unpack parameters
    (ny, nx), (dy, dx), phase_id, example_rotations = request.param

    # File columns
    map_size = ny * nx
    x = np.tile(np.arange(nx) * dx, ny)
    y = np.sort(np.tile(np.arange(ny) * dy, nx))
    iq = np.round(np.random.uniform(low=0, high=100, size=map_size), decimals=1)
    dp = np.round(np.random.random(map_size), decimals=3)  # [0, 1]

    # Rotations
    n_rotations = len(example_rotations)
    if n_rotations == map_size:
        rot = example_rotations
    else:
        # Sample as many rotations from `example_rotations` as `map_size`
        rot_idx = np.random.choice(np.arange(len(example_rotations)), map_size)
        rot = example_rotations[rot_idx]

    np.savetxt(
        fname=f,
        X=np.column_stack((rot[:, 0], rot[:, 1], rot[:, 2], x, y, iq, dp, phase_id)),
        fmt="%.5f %.5f %.5f %.5f %.5f %.1f %.3f %i",
        header=ANGFILE_EMSOFT_HEADER,
        comments="",
    )

    return f
