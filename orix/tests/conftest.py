# -*- coding: utf-8 -*-
# Copyright 2018-2023 the orix developers
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

import gc
import os
from tempfile import TemporaryDirectory

from diffpy.structure import Atom, Lattice, Structure
from h5py import File
import matplotlib.pyplot as plt
import numpy as np
import pytest

from orix.crystal_map import CrystalMap, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation


@pytest.fixture
def rotations():
    return Rotation([(2, 4, 6, 8), (-1, -3, -5, -7)])


@pytest.fixture()
def eu():
    return np.random.rand(10, 3)


ANGFILE_TSL_HEADER = (
    "# TEM_PIXperUM          1.000000\n"
    "# x-star                0.413900\n"
    "# y-star                0.729100\n"
    "# z-star                0.514900\n"
    "# WorkingDistance       27.100000\n"
    "#\n"
    "# Phase 2\n"
    "# MaterialName      Aluminum\n"
    "# Formula       Al\n"
    "# Info      \n"
    "# Symmetry              43\n"
    "# LatticeConstants      4.040 4.040 4.040  90.000  90.000  90.000\n"
    "# NumberFamilies        69\n"
    "# hklFamilies        1 -1 -1 1 8.469246 1\n"
    "# ElasticConstants  -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000\n"
    "# Categories0 0 0 0 0 \n"
    "# Phase 3\n"
    "# MaterialName  	Iron Titanium Oxide\n"
    "# Formula     	FeTiO3\n"
    "# Info 		\n"
    "# Symmetry              32\n"
    "# LatticeConstants      5.123 5.123 13.760  90.000  90.000 120.000\n"
    "# NumberFamilies        60\n"
    "# hklFamilies   	 3  0  0 1 100.000000 1\n"
    "# ElasticConstants  -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000\n"
    "# Categories0 0 0 0 0\n"
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


@pytest.fixture()
def temp_ang_file():
    with TemporaryDirectory() as tempdir:
        f = open(os.path.join(tempdir, "temp_ang_file.ang"), mode="w+")
        yield f
        gc.collect()  # Garbage collection so that file can be used by multiple tests


@pytest.fixture(params=["h5"])
def temp_file_path(request):
    """Temporary file in a temporary directory for use when tests need
    to write, and sometimes read again, data to, and from, a file.
    """
    ext = request.param
    with TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, "data_temp." + ext)
        yield file_path
        gc.collect()


@pytest.fixture(
    params=[
        # Tuple with default values for five parameters: map_shape, step_sizes,
        # phase_id, n_unknown_columns, and rotations (see docstring below)
        (
            (7, 13),  # map_shape
            (0.1, 0.1),  # step_sizes
            np.random.choice([2, 3], 7 * 13),  # phase_id
            5,  # Number of unknown columns (one between ci and fit, the rest after fit)
            np.array(
                [
                    [4.48549, 0.95242, 0.79150],
                    [1.34390, 0.27611, 0.82589],
                ]
            ),  # Rotations as rows of Euler angle triplets
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
    phase_id : numpy.ndarray
        Array of map size with phase IDs in header.
    n_unknown_columns : int
        Number of columns with values of unknown nature.
    rotations : numpy.ndarray
        A sample, smaller than the map size, of example rotations as
        rows of Euler angle triplets.
    """
    f = tmpdir.join("angfile_tsl.ang")

    # Unpack parameters
    (ny, nx), (dy, dx), phase_id, n_unknown_columns, example_rotations = request.param

    # File columns
    d, map_size = create_coordinate_arrays((ny, nx), (dy, dx))
    x = d["x"]
    y = d["y"]
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

    yield f
    gc.collect()


@pytest.fixture(
    params=[
        # Tuple with default values for four parameters: map_shape, step_sizes,
        # phase_id, and rotations (see docstring below)
        (
            (7, 5),  # map_shape
            (2.86, 2.86),  # step_sizes
            np.ones(7 * 5, dtype=int),  # phase_id
            np.array(
                [[6.148271, 0.792205, 1.324879], [6.155951, 0.793078, 1.325229]]
            ),  # rotations as rows of Euler angle triplets
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
    phase_id : numpy.ndarray
        Array of map size with phase IDs in header.
    rotations : numpy.ndarray
        A sample, smaller than the map size, of example rotations as
        rows of Euler angle triplets.
    """
    f = tmpdir.join("angfile_astar.ang")

    # Unpack parameters
    (ny, nx), (dy, dx), phase_id, example_rotations = request.param

    # File columns
    d, map_size = create_coordinate_arrays((ny, nx), (dy, dx))
    x = d["x"]
    y = d["y"]
    ind = np.random.uniform(low=0, high=100, size=map_size)
    rel = np.round(np.random.random(map_size), decimals=2)  # [0, 1]
    relx100 = (rel * 100).astype(int)

    # Rotations
    n_rotations = len(example_rotations)
    if n_rotations == map_size:
        rot = example_rotations
    else:
        # Sample as many rotations from `rotations` as `map_size`
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

    yield f
    gc.collect()


@pytest.fixture(
    params=[
        # Tuple with default values for four parameters: map_shape, step_sizes,
        # phase_id, and rotations (see docstring below)
        (
            (9, 7),  # map_shape
            (1.5, 1.5),  # step_sizes
            np.random.choice([1, 2], 9 * 7),  # phase_id
            np.array(
                [[6.148271, 0.792205, 1.324879], [6.155951, 0.793078, 1.325229]]
            ),  # rotations as rows of Euler angle triplets
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
    phase_id : numpy.ndarray
        Array of map size with phase IDs in header.
    rotations : numpy.ndarray
        A sample, smaller than the map size, of example rotations as
        rows of Euler angle triplets.
    """
    f = tmpdir.join("angfile_emsoft.ang")

    # Unpack parameters
    (ny, nx), (dy, dx), phase_id, example_rotations = request.param

    # File columns
    d, map_size = create_coordinate_arrays((ny, nx), (dy, dx))
    x = d["x"]
    y = d["y"]
    iq = np.round(np.random.uniform(low=0, high=100, size=map_size), decimals=1)
    dp = np.round(np.random.random(map_size), decimals=3)  # [0, 1]

    # Rotations
    rot_idx = np.random.choice(np.arange(len(example_rotations)), map_size)
    rot = example_rotations[rot_idx]

    np.savetxt(
        fname=f,
        X=np.column_stack((rot[:, 0], rot[:, 1], rot[:, 2], x, y, iq, dp, phase_id)),
        fmt="%.5f %.5f %.5f %.5f %.5f %.1f %.3f %i",
        header=ANGFILE_EMSOFT_HEADER,
        comments="",
    )

    yield f
    gc.collect()


@pytest.fixture(
    params=[
        # Tuple with default values for parameters: map_shape, step_sizes,
        # rotations, n_top_matches, and refined (see docstring below)
        (
            (13, 3),  # map_shape
            (1.5, 1.5),  # step_sizes
            np.array(
                [[6.148271, 0.792205, 1.324879], [6.155951, 0.793078, 1.325229]]
            ),  # rotations as rows of Euler angle triplets
            50,  # n_top_matches
            True,  # refined
        )
    ]
)
def temp_emsoft_h5ebsd_file(tmpdir, request):
    """Create a dummy EMsoft h5ebsd .h5 file from input.

    Parameters expected in `request`
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    step_sizes : tuple of floats
        Step sizes in x and y coordinates in microns.
    rotations : numpy.ndarray
        A sample, smaller than the map size, of example rotations as
        rows of Euler angle triplets.
    n_top_matches : int
        Number of top matching orientations per data point kept.
    refined : bool
        Whether refined Euler angles and dot products are read.
    """
    f = File(tmpdir.join("emsoft_h5ebsd_file.h5"), mode="w")

    # Unpack parameters
    map_shape, (dy, dx), example_rotations, n_top_matches, refined = request.param
    ny, nx = map_shape
    d, map_size = create_coordinate_arrays(map_shape, (dy, dx))

    # Create groups used in reader
    f.create_dataset(name="Manufacturer", data="EMEBSDDictionaryIndexing.f90")
    ebsd_group = f.create_group("Scan 1/EBSD")
    data_group = ebsd_group.create_group("Data")
    header_group = ebsd_group.create_group("Header")
    phase_group = header_group.create_group("Phase/1")  # Always single phase

    # Create `header_group` datasets used in reader
    for name, data, dtype in zip(
        ["nRows", "nColumns", "Step Y", "Step X"],
        [ny, nx, dy, dx],
        [np.int32, np.int32, np.float32, np.float32],
    ):
        header_group.create_dataset(name, data=np.array([data], dtype=dtype))

    # Create `data_group` datasets, mostly quality metrics
    data_group.create_dataset("X Position", data=d["x"])
    # Note that "Y Position" is wrongly written to their h5ebsd file by EMsoft
    data_group.create_dataset(
        "Y Position",
        data=np.tile(np.arange(nx) * dx, ny),  # Wrong
        # data=d["y"],  # Correct
    )
    for name, shape, dtype in [
        ("AvDotProductMap", map_shape, np.int32),
        ("CI", map_size, np.float32),
        ("IQ", map_size, np.float32),
        ("ISM", map_size, np.float32),
        ("KAM", map_shape, np.float32),
        ("OSM", map_shape, np.float32),
        ("Phase", map_size, np.uint8),
    ]:
        data_group.create_dataset(name, data=np.zeros(shape, dtype=dtype))

    # `data_group` with rotations
    # Sample as many rotations from `rotations` as `map_size`
    rot_idx = np.random.choice(np.arange(len(example_rotations)), map_size)
    rot = example_rotations[rot_idx]
    n_sampled_oris = 333227  # Cubic space group with Ncubochoric = 100
    data_group.create_dataset("FZcnt", data=np.array([n_sampled_oris], dtype=np.int32))
    data_group.create_dataset(
        "TopMatchIndices",
        data=np.vstack(
            (np.random.choice(np.arange(n_sampled_oris), n_top_matches),) * map_size
        ),
        dtype=np.int32,
    )
    data_group.create_dataset(
        "TopDotProductList",
        data=np.vstack((np.random.random(size=n_top_matches),) * map_size),
        dtype=np.float32,
    )
    # In degrees
    data_group.create_dataset(
        "DictionaryEulerAngles",
        data=np.column_stack((np.linspace(150, 160, n_sampled_oris),) * 3),
        dtype=np.float32,
    )

    if refined:
        data_group.create_dataset("RefinedEulerAngles", data=rot.astype(np.float32))
        data_group.create_dataset(
            "RefinedDotProducts", data=np.zeros(map_size, dtype=np.float32)
        )

    # Number of top matches kept
    f.create_dataset(
        "NMLparameters/EBSDIndexingNameListType/nnk",
        data=np.array([n_top_matches], dtype=np.int32),
    )

    # `phase_group`
    for name, data in [
        ("Point Group", "Monoclinic b (C2h) [2/m]"),
        ("MaterialName", "fe4al13/fe4al13"),
        ("Lattice Constant a", "15.009001"),
        ("Lattice Constant b", "8.066"),
        ("Lattice Constant c", "12.469"),
        ("Lattice Constant alpha", "90.0"),
        ("Lattice Constant beta", "107.72"),
        ("Lattice Constant gamma", "90.0"),
    ]:
        phase_group.create_dataset(name, data=np.array([data], dtype=np.dtype("S")))

    yield f
    gc.collect()


@pytest.fixture(
    params=[
        # Tuple with default values for parameters: map_shape,
        # step_sizes, phase_id, rotations, and whether to shuffle map
        # points in data arrays
        (
            (9, 7),  # map_shape
            (1.5, 1.5),  # step_sizes
            np.random.choice([1, 2], 9 * 7),  # phase_id
            np.array([[35, 75, 13], [14, 0, 26]]),  # rotations
            False,  # Whether to shuffle order of map points
        )
    ]
)
def temp_bruker_h5ebsd_file(tmpdir, request):
    """Create a dummy Bruker h5ebsd .h5 file from input.

    Parameters expected in `request`
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    step_sizes : tuple of floats
        Step sizes in x and y coordinates in microns.
    phase_id : numpy.ndarray
        Array of map size with phase IDs in header.
    rotations : numpy.ndarray
        A sample, smaller than the map size, of example rotations as
        rows of Euler angle triplets.
    """
    f = File(tmpdir.join("bruker_h5ebsd_file.h5"), mode="w")

    # Unpack parameters
    map_shape, (dy, dx), phase_id, example_rotations, shuffle_order = request.param
    ny, nx = map_shape
    map_rows, map_cols = np.indices(map_shape)
    map_rows = map_rows.ravel()
    map_cols = map_cols.ravel()
    y = map_rows * dy
    x = map_cols * dx
    map_size = ny * nx

    # Create groups used in reader
    f.create_dataset(name="Manufacturer", data=b"Bruker Nano")
    ebsd_group = f.create_group("Scan 1/EBSD")
    data_group = ebsd_group.create_group("Data")
    header_group = ebsd_group.create_group("Header")
    sem_group = ebsd_group.create_group("SEM")

    # Write phases
    phases_group = header_group.create_group("Phases")
    unique_phase_ids = np.unique(phase_id)
    characters = "abcdefghijklmnopqrstuvwzyx"
    for i, pid in enumerate(unique_phase_ids):
        phase_group = phases_group.create_group(str(pid))
        phase_group.create_dataset("Formula", data=characters[i])
        phase_group.create_dataset("IT", data=225)
        lattice_constants = np.array([i + 1] * 3 + [90] * 3)
        phase_group.create_dataset("LatticeConstants", data=lattice_constants)
        phase_group.create_dataset("Name", data=characters[i])
        phase_group.create_dataset("Setting", data=1)
        phase_group.create_dataset("SpaceGroup", data=b"F m#ovl3m")
        atom_positions = phase_group.create_group("AtomPositions")
        for k in range(3):
            atom_pos_str = f"{characters[k]},{k},{k},{k},1,0".encode()
            atom_positions.create_dataset(str(k), data=atom_pos_str)

    # Write SEM data
    if shuffle_order:
        rng = np.random.default_rng()
        # Only shuffle within rows (rows are not shuffled)
        map_cols = map_cols.reshape(map_shape)
        rng.shuffle(map_cols)
        map_cols = map_cols.ravel()
    sem_group.create_dataset("IY", data=map_rows)
    sem_group.create_dataset("IX", data=map_cols)

    rc = np.array([map_rows, map_cols])
    map_order = np.ravel_multi_index(rc, map_shape).argsort()

    # Write properties
    zeros_float = np.zeros(map_size, dtype=np.float32)
    zeros_int = zeros_float.astype(np.int32)
    for name, data in [
        ("DD", zeros_float),
        ("MAD", zeros_float),
        ("MADPhase", zeros_int),
        ("NIndexedBands", zeros_int),
        ("PCX", zeros_float),
        ("PCY", zeros_float),
        ("RadonBandCount", zeros_int),
        ("RadonQuality", zeros_float),
        ("Y BEAM", map_rows),
        ("X BEAM", map_cols),
        ("Y SAMPLE", y[map_order]),
        ("X SAMPLE", x[map_order][::-1]),  # Bruker flips x in file
        ("Z SAMPLE", zeros_int),
        ("Phase", phase_id[map_order]),
    ]:
        data_group.create_dataset(name, data=data)

    # Write header
    header_group.create_dataset("NROWS", data=ny, dtype=np.int32)
    header_group.create_dataset("NCOLS", data=nx, dtype=np.int32)
    header_group.create_dataset("Grid Type", data=b"isometric")

    # Write rotations
    rot_idx = np.random.choice(np.arange(len(example_rotations)), map_size)
    rot = example_rotations[rot_idx][map_order]
    data_group.create_dataset("phi1", data=rot[:, 0])
    data_group.create_dataset("PHI", data=rot[:, 1])
    data_group.create_dataset("phi2", data=rot[:, 2])

    yield f
    gc.collect()


@pytest.fixture(
    params=[
        (
            ["a", "b", "c"],
            [229, 207, 143],
            ["m-3m", "432", "3"],
            ["r", "g", "b"],
            [Lattice()] * 3,
            [[Atom()]] * 3,
        )
    ]
)
def phase_list(request):
    names, space_groups, point_group_names, colors, lattices, atoms = request.param
    # Apparently diffpy.structure don't allow iteration over a list of lattices
    structures = [Structure(lattice=lattices[i], atoms=a) for i, a in enumerate(atoms)]
    return PhaseList(
        names=names,
        space_groups=space_groups,
        point_groups=point_group_names,
        colors=colors,
        structures=structures,
    )


@pytest.fixture(
    params=[
        (
            # Tuple with default values for parameters: map_shape, step_sizes,
            # and n_rotations_per_point
            (4, 3),  # map_shape
            (1.5, 1.5),  # step_sizes
            1,  # rotations_per_point
            [0],  # unique phase IDs
        )
    ],
)
def crystal_map_input(request, rotations):
    # Unpack parameters
    (ny, nx), (dy, dx), rotations_per_point, unique_phase_ids = request.param
    d, map_size = create_coordinate_arrays((ny, nx), (dy, dx))
    rot_idx = np.random.choice(
        np.arange(rotations.size), map_size * rotations_per_point
    )
    data_shape = (map_size,)
    if rotations_per_point > 1:
        data_shape += (rotations_per_point,)
    d["rotations"] = rotations[rot_idx].reshape(*data_shape)
    phase_id = np.random.choice(unique_phase_ids, map_size)
    for i in range(len(unique_phase_ids)):
        phase_id[i] = unique_phase_ids[i]
    d["phase_id"] = phase_id
    return d


@pytest.fixture
def crystal_map(crystal_map_input):
    return CrystalMap(**crystal_map_input)


@pytest.fixture
def cif_file(tmpdir):
    """Actual CIF file of beta double prime phase often seen in Al-Mg-Si
    alloys.
    """
    file_contents = """
#======================================================================

# CRYSTAL DATA

#----------------------------------------------------------------------

data_VESTA_phase_1


_chemical_name_common                  ''
_cell_length_a                         15.50000
_cell_length_b                         4.05000
_cell_length_c                         6.74000
_cell_angle_alpha                      90
_cell_angle_beta                       105.30000
_cell_angle_gamma                      90
_space_group_name_H-M_alt              'C 2/m'
_space_group_IT_number                 12

loop_
_space_group_symop_operation_xyz
   'x, y, z'
   '-x, -y, -z'
   '-x, y, -z'
   'x, -y, z'
   'x+1/2, y+1/2, z'
   '-x+1/2, -y+1/2, -z'
   '-x+1/2, y+1/2, -z'
   'x+1/2, -y+1/2, z'

loop_
   _atom_site_label
   _atom_site_occupancy
   _atom_site_fract_x
   _atom_site_fract_y
   _atom_site_fract_z
   _atom_site_adp_type
   _atom_site_B_iso_or_equiv
   _atom_site_type_symbol
   Mg(1)      1.0     0.000000      0.000000      0.000000     Biso  1.000000 Mg
   Mg(2)      1.0     0.347000      0.000000      0.089000     Biso  1.000000 Mg
   Mg(3)      1.0     0.423000      0.000000      0.652000     Biso  1.000000 Mg
   Si(1)      1.0     0.054000      0.000000      0.649000     Biso  1.000000 Si
   Si(2)      1.0     0.190000      0.000000      0.224000     Biso  1.000000 Si
   Al         1.0     0.211000      0.000000      0.626000     Biso  1.000000 Al"
"""
    f = open(tmpdir.join("betapp.cif"), mode="w")
    f.write(file_contents)
    f.close()
    yield f.name
    gc.collect()


# ---------- Rotation representations for conversion tests ----------- #
# NOTE to future test writers on unittest data:
# All the data below can be recreated using 3Drotations, which is
# available at
# https://github.com/marcdegraef/3Drotations/blob/master/src/python.
# 3Drotations is an expanded implementation of the rotation conversions
# laid out in Rowenhorst et al. (2015), written by a subset of the
# original authors.
# Note, however, that orix differs from 3Drotations in its handling of
# some edge cases. Software using 3Drotations (for example, Dream3D and
# EBSDLib), handle rounding and other corrections after converting,
# whereas orix accounts for them during. The first three angles in each
# set are tests of these edge cases, and will therefore differ between
# orix and 3Drotations. For all other angles, the datasets can be
# recreated using a variation of:
#   np.around(rotlib.qu2{insert_new_representation_here}(qu), 4).
# This consistently gives results with four decimals of accuracy.


@pytest.fixture
def cubochoric_coordinates():
    # fmt: off
    return np.array(
        [
            [np.pi ** (2 / 3) / 2 + 1e-7,    1,    1],
            [                          0,    0,    0],
            [                     1.0725,    0,    0],
            [                          0,    0,    1],
            [                        0.1,  0.1,  0.2],
            [                        0.1,  0.1, -0.2],
            [                        0.5,  0.2,  0.1],
            [                       -0.5, -0.2,  0.1],
            [                        0.2,  0.5,  0.1],
            [                        0.2, -0.5,  0.1],
        ],
        dtype=np.float64,
    )
    # fmt: on


@pytest.fixture
def homochoric_vectors():
    # fmt: off
    return np.array(
        [
            [      0,       0,       0],
            [      0,       0,       0],
            [ 1.3307,       0,       0],
            [      0,       0,  1.2407],
            [ 0.0785,  0.0785,  0.2219],
            [ 0.0785,  0.0785, -0.2219],
            [ 0.5879,  0.1801,  0.0827],
            [-0.5879, -0.1801,  0.0827],
            [ 0.1801,  0.5879,  0.0827],
            [ 0.1801, -0.5879,  0.0827],
        ],
        dtype=np.float64,
    )
    # fmt: on


@pytest.fixture
def axis_angle_pairs():
    # fmt: off
    return np.array(
        [
            [      0,       0,       1,      0],
            [      0,       0,       1,      0],
            [      1,       0,       0,  np.pi],
            [      0,       0,       1, 2.8418],
            [ 0.3164,  0.3164,  0.8943, 0.4983],
            [ 0.3164,  0.3164, -0.8943, 0.4983],
            [ 0.9476,  0.2903,  0.1333, 1.2749],
            [-0.9476, -0.2903,  0.1333, 1.2749],
            [ 0.2903,  0.9476,  0.1333, 1.2749],
            [ 0.2903, -0.9476,  0.1333, 1.2749],
        ],
        dtype=np.float64,
    )
    # fmt: on


@pytest.fixture
def rodrigues_vectors():
    # fmt: off
    return np.array(
        [
            [      0,       0,       1,      0],
            [      0,       0,       1,      0],
            [      1,       0,       0, np.inf],
            [      0,       0,       1, 6.6212],
            [ 0.3164,  0.3164,  0.8943, 0.2544],
            [ 0.3164,  0.3164, -0.8943, 0.2544],
            [ 0.9476,  0.2903,  0.1333, 0.7406],
            [-0.9476, -0.2903,  0.1333, 0.7406],
            [ 0.2903,  0.9476,  0.1333, 0.7406],
            [ 0.2903, -0.9476,  0.1333, 0.7406]
        ],
        dtype=np.float64,
    )
    # fmt: on


@pytest.fixture
def orientation_matrices():
    # fmt: off
    return np.array(
        [
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            [
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1],
            ],
            [
                [-0.9554, -0.2953, 0],
                [ 0.2953, -0.9554, 0],
                [      0,       0, 1],
            ],
            [
                [ 0.8906, -0.4152,  0.1856],
                [ 0.4396,  0.8906, -0.1168],
                [-0.1168,  0.1856,  0.9757],
            ],
            [
                [ 0.8906, 0.4396,  0.1168],
                [-0.4152, 0.8906, -0.1856],
                [-0.1856, 0.1168,  0.9757],
            ],
            [
                [ 0.9277, 0.0675,  0.3672],
                [ 0.3224, 0.3512, -0.879 ],
                [-0.1883, 0.9339,  0.3041],
            ],
            [
                [0.9277,  0.0675, -0.3672],
                [0.3224,  0.3512,  0.879 ],
                [0.1883, -0.9339,  0.3041],
            ],
            [
                [ 0.3512, 0.0675,  0.9339],
                [ 0.3224, 0.9277, -0.1883],
                [-0.879 , 0.3672,  0.3041],
            ],
            [
                [ 0.3512, -0.3224, -0.879 ],
                [-0.0675,  0.9277, -0.3672],
                [ 0.9339,  0.1883,  0.3041],
            ],
        ],
        dtype=np.float64
    )
    # fmt: on


@pytest.fixture
def quaternions_conversions():
    # fmt: off
    return np.array(
        [
            [     1,       0,       0,       0],
            [     1,       0,       0,       0],
            [     0,       1,       0,       0],
            [0.1493,       0,       0,  0.9888],
            [0.9691,  0.0780,  0.0780,  0.2205],
            [0.9691,  0.0780,  0.0780, -0.2205],
            [0.8036,  0.5640,  0.1728,  0.0793],
            [0.8036, -0.5640, -0.1728,  0.0793],
            [0.8036,  0.1728,  0.5640,  0.0793],
            [0.8036,  0.1728, -0.5640,  0.0793],
        ],
        dtype=np.float64,
    )
    # fmt: on


@pytest.fixture
def euler_angles():
    # fmt: off
    return np.array(
        [
            [     0,      0,      0],
            [     0,      0,      0],
            [     0, 3.1416,      0],
            [3.4413,      0,      0],
            [3.7033, 0.2211, 2.1325],
            [4.1507, 0.2211, 2.5799],
            [3.3405, 1.2618, 2.7459],
            [0.1989, 1.2618, 5.8875],
            [4.3167, 1.2618, 1.7697],
            [1.7697, 1.2618, 4.3167],
        ],
        dtype=np.float64,
    )
    # fmt: on


# ------- End of rotation representations for conversion tests ------- #


@pytest.fixture(autouse=True)
def import_to_namespace(doctest_namespace):
    """Make :mod:`numpy` and :mod:`matplotlib.pyplot` available in
    docstring examples without having to import them.

    See https://docs.pytest.org/en/stable/how-to/doctest.html#doctest-namespace-fixture.
    """
    doctest_namespace["plt"] = plt
    doctest_namespace["np"] = np
