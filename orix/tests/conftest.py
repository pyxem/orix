# Copyright 2018-2024 the orix developers
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
from tempfile import TemporaryDirectory

from diffpy.structure import Atom, Lattice, Structure
from h5py import File
import matplotlib.pyplot as plt
import numpy as np
import pytest

from orix import constants
from orix.crystal_map import CrystalMap, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation

# --------------------------- pytest hooks --------------------------- #


def pytest_sessionstart(session):  # pragma: no cover
    plt.rcParams["backend"] = "agg"


# -------------------- Control of test selection --------------------- #

skipif_numpy_quaternion_present = pytest.mark.skipif(
    constants.installed["numpy-quaternion"], reason="numpy-quaternion installed"
)

skipif_numpy_quaternion_missing = pytest.mark.skipif(
    not constants.installed["numpy-quaternion"], reason="numpy-quaternion not installed"
)

# ---------------------------- IO fixtures --------------------------- #

# ----------------------------- .ang file ---------------------------- #


@pytest.fixture()
def temp_ang_file():
    with TemporaryDirectory() as tempdir:
        f = open(os.path.join(tempdir, "temp_ang_file.ang"), mode="w+")
        yield f


@pytest.fixture(params=["h5"])
def temp_file_path(request):
    """Temporary file in a temporary directory for use when tests need
    to write, and sometimes read again, data to, and from, a file.
    """
    ext = request.param
    with TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, "data_temp." + ext)
        yield file_path


ANGFILE_TSL_HEADER = r"""# TEM_PIXperUM          1.000000
# x-star                0.413900
# y-star                0.729100
# z-star                0.514900
# WorkingDistance       27.100000
#
# Phase 2
# MaterialName      Aluminum
# Formula       Al
# Info
# Symmetry              43
# LatticeConstants      4.040 4.040 4.040  90.000  90.000  90.000
# NumberFamilies        69
# hklFamilies        1 -1 -1 1 8.469246 1
# ElasticConstants  -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000
# Categories0 0 0 0 0
# Phase 3
# MaterialName  	Iron Titanium Oxide
# Formula     	FeTiO3
# Info
# Symmetry              32
# LatticeConstants      5.123 5.123 13.760  90.000  90.000 120.000
# NumberFamilies        60
# hklFamilies   	 3  0  0 1 100.000000 1
# ElasticConstants  -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000
# Categories0 0 0 0 0
#
# GRID: SqrGrid
# XSTEP: 0.100000
# YSTEP: 0.100000
# NCOLS_ODD: 42
# NCOLS_EVEN: 42
# NROWS: 13
#
# OPERATOR:     sem
#
# SAMPLEID:
#
# SCANID:
#"""


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


ANGFILE_ASTAR_HEADER = r"""# File created from ACOM RES results
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


ANGFILE_EMSOFT_HEADER = r"""# TEM_PIXperUM          1.000000
# x-star                 0.446667
# y-star                 0.586875
# z-star                 0.713450
# WorkingDistance        0.000000
#
# Phase 1
# MaterialName    austenite
# Formula       austenite
# Info          patterns indexed using EMsoft::EMEBSDDI
# Symmetry              43
# LatticeConstants      3.595 3.595 3.595   90.000 90.000 90.000
# NumberFamilies        0
# Phase 2
# MaterialName    ferrite/ferrite
# Formula       ferrite/ferrite
# Info          patterns indexed using EMsoft::EMEBSDDI
# Symmetry              43
# LatticeConstants      2.867 2.867 2.867   90.000 90.000 90.000
# NumberFamilies        0
# GRID: SqrGrid
# XSTEP:  1.500000
# YSTEP:  1.500000
# NCOLS_ODD:   13
# NCOLS_EVEN:   13
# NROWS:   42
#
# OPERATOR:   Håkon Wiik Ånes
#
# SAMPLEID:
#
# SCANID:
#"""


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


# ----------------------------- .ctf file ---------------------------- #

# Variable map shape and step sizes
CTF_OXFORD_HEADER = r"""Channel Text File
Prj	standard steel sample
Author	
JobMode	Grid
XCells	%i
YCells	%i
XStep	%.4f
YStep	%.4f
AcqE1	0.0000
AcqE2	0.0000
AcqE3	0.0000
Euler angles refer to Sample Coordinate system (CS0)!	Mag	180.0000	Coverage	97	Device	0	KV	20.0000	TiltAngle	70.0010	TiltAxis	0	DetectorOrientationE1	0.9743	DetectorOrientationE2	89.4698	DetectorOrientationE3	2.7906	WorkingDistance	14.9080	InsertionDistance	185.0
Phases	2
3.660;3.660;3.660	90.000;90.000;90.000	Iron fcc	11	225			Some reference
2.867;2.867;2.867	90.000;90.000;90.000	Iron bcc	11	229			Some other reference
Phase	X	Y	Bands	Error	Euler1	Euler2	Euler3	MAD	BC	BS"""


@pytest.fixture(
    params=[
        (
            (7, 13),  # map_shape
            (0.1, 0.1),  # step_sizes
            np.random.choice([1, 2], 7 * 13),  # phase_id
            np.array([[4.48549, 0.95242, 0.79150], [1.34390, 0.27611, 0.82589]]),  # R
        )
    ]
)
def ctf_oxford(tmpdir, request):
    """Create a dummy CTF file in Oxford Instrument's format from input.

    10% of points are non-indexed (phase ID of 0 and MAD = 0).

    Parameters expected in `request`
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    step_sizes : tuple of floats
        Step sizes in x and y coordinates in microns.
    phase_id : numpy.ndarray
        Array of map size with phase IDs in header.
    rotations : numpy.ndarray
        A sample, smaller than the map size, of Euler angle triplets.
    """
    # Unpack parameters
    (ny, nx), (dy, dx), phase_id, R_example = request.param

    # File columns
    d, map_size = create_coordinate_arrays((ny, nx), (dy, dx))
    x, y = d["x"], d["y"]
    rng = np.random.default_rng()
    bands = rng.integers(8, size=map_size, dtype=np.uint8)
    err = np.zeros(map_size, dtype=np.uint8)
    mad = rng.random(map_size)
    bc = rng.integers(150, 200, map_size)
    bs = rng.integers(190, 255, map_size)
    R_idx = np.random.choice(np.arange(len(R_example)), map_size)
    R = R_example[R_idx]
    R = np.rad2deg(R)

    # Insert 10% non-indexed points
    non_indexed_points = np.random.choice(
        np.arange(map_size), replace=False, size=int(map_size * 0.1)
    )
    phase_id[non_indexed_points] = 0
    R[non_indexed_points] = 0.0
    bands[non_indexed_points] = 0
    err[non_indexed_points] = 3
    mad[non_indexed_points] = 0.0
    bc[non_indexed_points] = 0
    bs[non_indexed_points] = 0

    CTF_OXFORD_HEADER2 = CTF_OXFORD_HEADER % (nx, ny, dx, dy)

    f = tmpdir.join("oxford.ctf")
    np.savetxt(
        fname=f,
        X=np.column_stack(
            (phase_id, x, y, bands, err, R[:, 0], R[:, 1], R[:, 2], mad, bc, bs)
        ),
        fmt="%-4i%-8.4f%-8.4f%-4i%-4i%-11.4f%-11.4f%-11.4f%-8.4f%-4i%-i",
        header=CTF_OXFORD_HEADER2,
        comments="",
    )

    yield f


# Variable map shape, comma as decimal separator in fixed step size
CTF_BRUKER_HEADER = r"""Channel Text File
Prj unnamed
Author	[Unknown]
JobMode	Grid
XCells	%i
YCells	%i
XStep	0,001998
YStep	0,001998
AcqE1	0
AcqE2	0
AcqE3	0
Euler angles refer to Sample Coordinate system (CS0)!	Mag	150000,000000	Coverage	100	Device	0	KV	30,000000	TiltAngle	0	TiltAxis	0
Phases	1
4,079000;4,079000;4,079000	90,000000;90,000000;90,000000	Gold	11	225
Phase	X	Y	Bands	Error	Euler1	Euler2	Euler3	MAD	BC	BS"""


@pytest.fixture(
    params=[
        (
            (7, 13),  # map_shape
            np.array([[4.48549, 0.95242, 0.79150], [1.34390, 0.27611, 0.82589]]),  # R
        )
    ]
)
def ctf_bruker(tmpdir, request):
    """Create a dummy CTF file in Bruker's format from input.

    Identical to Oxford files except for the following:

    * All band slopes (BS) may be set to 255
    * Decimal separators in header may be with comma

    Parameters expected in `request`
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    rotations : numpy.ndarray
        A sample, smaller than the map size, of Euler angle triplets.
    """
    # Unpack parameters
    (ny, nx), R_example = request.param
    dy = dx = 0.001998

    # File columns
    d, map_size = create_coordinate_arrays((ny, nx), (dy, dx))
    x, y = d["x"], d["y"]
    rng = np.random.default_rng()
    bands = rng.integers(8, size=map_size, dtype=np.uint8)
    err = np.zeros(map_size, dtype=np.uint8)
    mad = rng.random(map_size)
    bc = rng.integers(50, 105, map_size)
    bs = np.full(map_size, 255, dtype=np.uint8)
    R_idx = np.random.choice(np.arange(len(R_example)), map_size)
    R = R_example[R_idx]
    R = np.rad2deg(R)

    # Insert 10% non-indexed points
    phase_id = np.ones(map_size, dtype=np.uint8)
    non_indexed_points = np.random.choice(
        np.arange(map_size), replace=False, size=int(map_size * 0.1)
    )
    phase_id[non_indexed_points] = 0
    R[non_indexed_points] = 0.0
    bands[non_indexed_points] = 0
    err[non_indexed_points] = 3
    mad[non_indexed_points] = 0.0
    bc[non_indexed_points] = 0
    bs[non_indexed_points] = 0

    CTF_BRUKER_HEADER2 = CTF_BRUKER_HEADER % (nx, ny)

    f = tmpdir.join("bruker.ctf")
    np.savetxt(
        fname=f,
        X=np.column_stack(
            (phase_id, x, y, bands, err, R[:, 0], R[:, 1], R[:, 2], mad, bc, bs)
        ),
        fmt="%-4i%-8.4f%-8.4f%-4i%-4i%-11.4f%-11.4f%-11.4f%-8.4f%-4i%-i",
        header=CTF_BRUKER_HEADER2,
        comments="",
    )

    yield f


# Variable map shape, small fixed step size
CTF_ASTAR_HEADER = r"""Channel Text File
Prj	C:\some\where\scan.res
Author	File created from ACOM RES results
JobMode	Grid
XCells	%i
YCells	%i
XStep	0.00191999995708466
YStep	0.00191999995708466
AcqE1	0
AcqE2	0
AcqE3	0
Euler angles refer to Sample Coordinate system (CS0)!	Mag	200	Coverage	100	Device	0	KV	20	TiltAngle	70	TiltAxis	0
Phases	1
4.0780;4.0780;4.0780	90;90;90	_mineral 'Gold'  'Gold'	11	225
Phase	X	Y	Bands	Error	Euler1	Euler2	Euler3	MAD	BC	BS"""


@pytest.fixture(
    params=[
        (
            (7, 13),  # map_shape
            np.array([[4.48549, 0.95242, 0.79150], [1.34390, 0.27611, 0.82589]]),  # R
        )
    ]
)
def ctf_astar(tmpdir, request):
    """Create a dummy CTF file in NanoMegas ASTAR's format from input.

    Identical to Oxford files except for the following:

    * Bands = 6 (always?)
    * Error = 0 (always?)
    * Only two decimals in Euler angles

    Parameters expected in `request`
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    rotations : numpy.ndarray
        A sample, smaller than the map size, of Euler angle triplets.
    """
    # Unpack parameters
    (ny, nx), R_example = request.param
    dy = dx = 0.00191999995708466

    # File columns
    d, map_size = create_coordinate_arrays((ny, nx), (dy, dx))
    x, y = d["x"], d["y"]
    rng = np.random.default_rng()
    bands = np.full(map_size, 6, dtype=np.uint8)
    err = np.zeros(map_size, dtype=np.uint8)
    mad = rng.random(map_size)
    bc = rng.integers(0, 60, map_size)
    bs = rng.integers(35, 42, map_size)
    R_idx = np.random.choice(np.arange(len(R_example)), map_size)
    R = R_example[R_idx]
    R = np.rad2deg(R)

    # Insert 10% non-indexed points
    phase_id = np.ones(map_size, dtype=np.uint8)
    non_indexed_points = np.random.choice(
        np.arange(map_size), replace=False, size=int(map_size * 0.1)
    )
    phase_id[non_indexed_points] = 0
    R[non_indexed_points] = 0.0
    bands[non_indexed_points] = 0
    err[non_indexed_points] = 3
    mad[non_indexed_points] = 0.0
    bc[non_indexed_points] = 0
    bs[non_indexed_points] = 0

    CTF_ASTAR_HEADER2 = CTF_ASTAR_HEADER % (nx, ny)

    f = tmpdir.join("astar.ctf")
    np.savetxt(
        fname=f,
        X=np.column_stack(
            (phase_id, x, y, bands, err, R[:, 0], R[:, 1], R[:, 2], mad, bc, bs)
        ),
        fmt="%-4i%-8.4f%-8.4f%-4i%-4i%-9.2f%-9.2f%-9.2f%-8.4f%-4i%-i",
        header=CTF_ASTAR_HEADER2,
        comments="",
    )

    yield f


# Variable map shape and step sizes
CTF_EMSOFT_HEADER = r"""Channel Text File
EMsoft v. 4_1_1_9d5269a; BANDS=pattern index, MAD=CI, BC=OSM, BS=IQ
Author	Me
JobMode	Grid
XCells	  %i
YCells	  %i
XStep	  %.2f
YStep	  %.2f
AcqE1	0
AcqE2	0
AcqE3	0
Euler angles refer to Sample Coordinate system (CS0)!	Mag	30	Coverage	100	Device	0	KV	 0.0	TiltAngle	0.00	TiltAxis	0
Phases	1
3.524;3.524;3.524	90.000;90.000;90.000	Ni	11	225
Phase	X	Y	Bands	Error	Euler1	Euler2	Euler3	MAD	BC	BS"""


@pytest.fixture(
    params=[
        (
            (7, 13),  # map_shape
            (1, 2),  # step_sizes
            np.array([[4.48549, 0.95242, 0.79150], [1.34390, 0.27611, 0.82589]]),  # R
        )
    ]
)
def ctf_emsoft(tmpdir, request):
    """Create a dummy CTF file in EMsoft's format from input.

    Identical to Oxford files except for the following:

    * Bands = dictionary index
    * Error = 0
    * Only three decimals in Euler angles

    Parameters expected in `request`
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    step_sizes : tuple of floats
        Step sizes in x and y coordinates in microns.
    rotations : numpy.ndarray
        A sample, smaller than the map size, of Euler angle triplets.
    """
    # Unpack parameters
    (ny, nx), (dy, dx), R_example = request.param

    # File columns
    d, map_size = create_coordinate_arrays((ny, nx), (dy, dx))
    x, y = d["x"], d["y"]
    rng = np.random.default_rng()
    bands = rng.integers(0, 333_000, map_size)
    err = np.zeros(map_size, dtype=np.uint8)
    mad = rng.random(map_size)
    bc = rng.integers(60, 140, map_size)
    bs = rng.integers(60, 120, map_size)
    R_idx = np.random.choice(np.arange(len(R_example)), map_size)
    R = R_example[R_idx]
    R = np.rad2deg(R)

    # Insert 10% non-indexed points
    phase_id = np.ones(map_size, dtype=np.uint8)
    non_indexed_points = np.random.choice(
        np.arange(map_size), replace=False, size=int(map_size * 0.1)
    )
    phase_id[non_indexed_points] = 0
    R[non_indexed_points] = 0.0
    bands[non_indexed_points] = 0
    mad[non_indexed_points] = 0.0
    bc[non_indexed_points] = 0
    bs[non_indexed_points] = 0

    CTF_EMSOFT_HEADER2 = CTF_EMSOFT_HEADER % (nx, ny, dx, dy)

    f = tmpdir.join("emsoft.ctf")
    np.savetxt(
        fname=f,
        X=np.column_stack(
            (phase_id, x, y, bands, err, R[:, 0], R[:, 1], R[:, 2], mad, bc, bs)
        ),
        fmt="%-4i%-8.4f%-8.4f%-7i%-4i%-10.3f%-10.3f%-10.3f%-8.4f%-4i%-i",
        header=CTF_EMSOFT_HEADER2,
        comments="",
    )

    yield f


# Variable map shape and step sizes
CTF_MTEX_HEADER = r"""Channel Text File
Prj /some/where/mtex.ctf
Author	Me Again
JobMode	Grid
XCells	%i
YCells	%i
XStep	%.4f
YStep	%.4f
AcqE1	0.0000
AcqE2	0.0000
AcqE3	0.0000
Euler angles refer to Sample Coordinate system (CS0)!	Mag	0.0000	Coverage	0	Device	0	KV	0.0000	TiltAngle	0.0000	TiltAxis	0	DetectorOrientationE1	0.0000	DetectorOrientationE2	0.0000	DetectorOrientationE3	0.0000	WorkingDistance	0.0000	InsertionDistance	0.0000	
Phases	1
4.079;4.079;4.079	90.000;90.000;90.000	Gold	11	0			Created from mtex
Phase	X	Y	Bands	Error	Euler1	Euler2	Euler3	MAD	BC	BS"""


@pytest.fixture(
    params=[
        (
            (7, 13),  # map_shape
            (1, 2),  # step_sizes
            np.array([[4.48549, 0.95242, 0.79150], [1.34390, 0.27611, 0.82589]]),  # R
        )
    ]
)
def ctf_mtex(tmpdir, request):
    """Create a dummy CTF file in MTEX's format from input.

    Identical to Oxford files except for the properties Bands, Error,
    MAD, BC, and BS are all equal to 0.

    Parameters expected in `request`
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    step_sizes : tuple of floats
        Step sizes in x and y coordinates in microns.
    rotations : numpy.ndarray
        A sample, smaller than the map size, of Euler angle triplets.
    """
    # Unpack parameters
    (ny, nx), (dy, dx), R_example = request.param

    # File columns
    d, map_size = create_coordinate_arrays((ny, nx), (dy, dx))
    x, y = d["x"], d["y"]
    bands = np.zeros(map_size)
    err = np.zeros(map_size)
    mad = np.zeros(map_size)
    bc = np.zeros(map_size)
    bs = np.zeros(map_size)
    R_idx = np.random.choice(np.arange(len(R_example)), map_size)
    R = R_example[R_idx]
    R = np.rad2deg(R)

    # Insert 10% non-indexed points
    phase_id = np.ones(map_size, dtype=np.uint8)
    non_indexed_points = np.random.choice(
        np.arange(map_size), replace=False, size=int(map_size * 0.1)
    )
    phase_id[non_indexed_points] = 0
    R[non_indexed_points] = 0.0

    CTF_MTEX_HEADER2 = CTF_MTEX_HEADER % (nx, ny, dx, dy)

    f = tmpdir.join("mtex.ctf")
    np.savetxt(
        fname=f,
        X=np.column_stack(
            (phase_id, x, y, bands, err, R[:, 0], R[:, 1], R[:, 2], mad, bc, bs)
        ),
        fmt="%-4i%-8.4f%-8.4f%-4i%-4i%-11.4f%-11.4f%-11.4f%-8.4f%-4i%-i",
        header=CTF_MTEX_HEADER2,
        comments="",
    )

    yield f


# ---------------------------- HDF5 files ---------------------------- #


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


# --------------------------- Other files ---------------------------- #


@pytest.fixture
def cif_file(tmpdir):
    """Actual CIF file of beta double prime phase often seen in Al-Mg-Si
    alloys.
    """
    file_contents = """#======================================================================

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
   Al         1.0     0.211000      0.000000      0.626000     Biso  1.000000 Al"""
    f = open(tmpdir.join("betapp.cif"), mode="w")
    f.write(file_contents)
    f.close()
    yield f.name


# ----------------------- Crystal map fixtures ----------------------- #


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
    d["rotations"] = rotations[rot_idx].reshape(data_shape)
    phase_id = np.random.choice(unique_phase_ids, map_size)
    for i in range(len(unique_phase_ids)):
        phase_id[i] = unique_phase_ids[i]
    d["phase_id"] = phase_id
    return d


@pytest.fixture
def crystal_map(crystal_map_input):
    return CrystalMap(**crystal_map_input)


# ---------- Rotation representations for conversion tests ----------- #
# NOTE: to future test writers on unittest data:
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


# ------------------------ Geometry fixtures ------------------------- #


@pytest.fixture
def rotations():
    return Rotation([(2, 4, 6, 8), (-1, -3, -5, -7)])


@pytest.fixture()
def eu():
    return np.random.rand(10, 3)


@pytest.fixture(autouse=True)
def import_to_namespace(doctest_namespace):
    """Make :mod:`numpy` and :mod:`matplotlib.pyplot` available in
    docstring examples without having to import them.

    See https://docs.pytest.org/en/stable/how-to/doctest.html#doctest-namespace-fixture.
    """
    doctest_namespace["plt"] = plt
    doctest_namespace["np"] = np
