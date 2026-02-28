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

"""Reader of a crystal map from a file in the Channel Text File (CTF)
format.
"""

from io import TextIOWrapper
import re
from typing import Any, Literal

from diffpy.structure import Lattice, Structure
import numpy as np

from orix.crystal_map._phase_list import PhaseList
from orix.crystal_map.crystal_map import CrystalMap, _data_slices_from_coordinates
from orix.quaternion.rotation import Rotation

__all__ = ["file_reader"]

# Plugin description
format_name = "ctf"
file_extensions = ["ctf"]
writes = False
writes_this = None


def file_reader(filename: str) -> CrystalMap:
    """Return a crystal map from a file in the Channel Text File (CTF)
    format.

    The map in the input is assumed to be 2D.

    Many vendors/programs can write a .ctf file. Files from the
    following vendors/programs are tested:

    * Oxford Instruments AZtec
    * Bruker Esprit
    * NanoMegas ASTAR Index
    * EMsoft (from program `EMdpmerge`)
    * MTEX

    All points with a phase of 0 are classified as not indexed.

    Parameters
    ----------
    filename
        Path to file.

    Returns
    -------
    xmap
        Crystal map.

    Notes
    -----
    Files written by MTEX do not contain information of the space group.

    Files written by EMsoft have the column names for mean angular
    deviation (MAD), band contrast (BC), and band slope (BS) renamed to
    DP (dot product), OSM (orientation similarity metric), and IQ (image
    quality), respectively.

    Description of error codes provided in CTF file:
    - 0: Success
    - 1: Low BC
    - 2: Low BS
    - 3: No solution
    - 4: High MAD
    - 5: Not yet analyzed (job cancelled before point)
    - 6: Unexpected error (exceptions etc.)
    """
    with open(filename, "r") as f:
        header, data_starting_row, vendor = _get_header(f)

    # Phase information, potentially empty
    phases = _get_phases_from_header(header)
    phases["structures"] = []
    lattice_constants = phases.pop("lattice_constants")
    for name, abcABG in zip(phases["names"], lattice_constants):
        structure = Structure(title=name, lattice=Lattice(*abcABG))
        phases["structures"].append(structure)

    file_data = np.loadtxt(filename, skiprows=data_starting_row)

    # Data needed to create a crystal map
    data_dict = {
        "euler1": None,
        "euler2": None,
        "euler3": None,
        "x": None,
        "y": None,
        "phase_id": None,
        "prop": {},
    }
    column_names = [
        "phase_id",
        "x",
        "y",
        "bands",
        "error",
        "euler1",
        "euler2",
        "euler3",
        "MAD",  # Mean angular deviation
        "BC",  # Band contrast
        "BS",  # Band slope
    ]
    emsoft_mapping = {"MAD": "DP", "BC": "OSM", "BS": "IQ"}
    for column, name in enumerate(column_names):
        if name in data_dict:
            data_dict[name] = file_data[:, column]
        else:
            if vendor == "emsoft" and name in emsoft_mapping:
                name = emsoft_mapping[name]
            data_dict["prop"][name] = file_data[:, column]

    if vendor == "astar":
        data_dict = _fix_astar_coords(header, data_dict)

    data_dict["phase_list"] = PhaseList(**phases)

    # Set which data points are not indexed
    not_indexed = data_dict["phase_id"] == 0
    data_dict["phase_id"][not_indexed] = -1

    # Set scan unit
    data_dict["scan_unit"] = "um"

    # Create rotations
    data_dict["rotations"] = Rotation.from_euler(
        np.column_stack(
            (data_dict.pop("euler1"), data_dict.pop("euler2"), data_dict.pop("euler3"))
        ),
        degrees=True,
    )

    return CrystalMap(**data_dict)


def _get_header(
    file: TextIOWrapper,
) -> tuple[list[str], int, Literal["oxford_or_bruker", "emsoft", "astar", "mtex"]]:
    """Return file header, row number for the first line with data, and
    the detected vendor.

    Parameters
    ----------
    file
        File object.

    Returns
    -------
    header
        List with header lines.
    data_starting_row
        Row number for the first line with data.
    vendor
        Detected vendor. Default is to assume Oxford or Bruker,
        "oxford_or_bruker" (assuming identical CTF formatting). Other
        options are "emsoft", "astar", and "mtex".
    """
    vendor = []
    vendor_patterns = {
        "emsoft": re.compile(
            (
                r"EMsoft v\. ([A-Za-z0-9]+(_[A-Za-z0-9]+)+); BANDS=pattern index, "
                r"MAD=CI, BC=OSM, BS=IQ"
            ),
        ),
        "astar": re.compile(r"Author[\t\s]File created from ACOM RES results"),
        "mtex": re.compile("(?<=)Created from mtex"),
    }

    # Keep header lines and any matching vendor patterns (potentially
    # more than one)
    header = []
    line = file.readline()
    data_starting_row = 0
    max_header_lines = 1_000
    while data_starting_row < max_header_lines and not line.startswith("Phase\tX\tY"):
        for k, v in vendor_patterns.items():
            match = v.search(line)
            if match is not None:
                vendor.append(k)
        header.append(line.rstrip())
        data_starting_row += 1
        line = file.readline()

    if len(vendor) == 1:
        vendor = vendor[0]
    else:
        vendor = "oxford_or_bruker"

    return header, data_starting_row + 1, vendor


def _get_phases_from_header(header: list[str]) -> dict[str, list]:
    """Return phase names and symmetries detected in a CTF file header.

    Parameters
    ----------
    header
        List with header lines.

    Returns
    -------
    phases_dict
        Dictionary with the following keys (and types): "ids" (int),
        "names" (str), "space_groups" (int), "point_groups" (str),
        "lattice_constants" (list of floats).

    Notes
    -----
    This function has been tested with files from the following vendor's
    formats: Oxford AZtec HKL v5/v6 and EMsoft v4/v5.
    """
    phases_dict = {
        "ids": [],
        "names": [],
        "space_groups": [],
        "point_groups": [],
        "lattice_constants": [],
    }
    for line_number, line in enumerate(header):
        if line.startswith("Phases"):
            break

    n_phases = int(line.split("\t")[1])

    laue_ids = {
        1: "-1",
        2: "2/m",
        3: "mmm",
        4: "4/m",
        5: "4/mmm",
        6: "-3",
        7: "-3m",
        8: "6/m",
        9: "6/mmm",
        10: "m3",
        11: "m-3m",
    }

    for i in range(n_phases):
        phase_data = header[line_number + 1 + i].split("\t")
        phases_dict["ids"].append(i + 1)
        abcABG = ";".join(phase_data[:2])
        abcABG = abcABG.split(";")
        abcABG = [float(lat.replace(",", ".")) for lat in abcABG]
        phases_dict["lattice_constants"].append(abcABG)
        phases_dict["names"].append(phase_data[2])
        laue_id = int(phase_data[3])
        phases_dict["point_groups"].append(laue_ids[laue_id])
        sg = int(phase_data[4])
        if sg == 0:
            sg = None
        phases_dict["space_groups"].append(sg)

    return phases_dict


def _fix_astar_coords(header: list[str], data_dict: dict[str, Any]) -> dict[str, Any]:
    """Return the data dictionary with coordinate arrays possibly fixed
    for ASTAR Index files.

    Parameters
    ----------
    header
        List with header lines.
    data_dict
        Dictionary for creating a crystal map.

    Returns
    -------
    data_dict
        Dictionary with possibly fixed coordinate arrays.

    Notes
    -----
    ASTAR Index files may have fewer decimals in the coordinate columns
    than in the X/YSteps header values (e.g. X_1 = 0.0019 vs.
    XStep = 0.00191999995708466). This may cause our crystal map
    algorithm for finding the map shape to fail. We therefore run this
    algorithm and compare the found shape to the shape given in the
    file. If they are different, we use our own coordinate arrays.
    """
    coords = {k: data_dict[k] for k in ["x", "y"]}
    slices = _data_slices_from_coordinates(coords)
    found_shape = (slices[0].stop + 1, slices[1].stop + 1)
    cells = _get_xy_cells(header)
    shape = (cells["y"], cells["x"])
    if found_shape != shape:
        steps = _get_xy_step(header)
        y, x = np.indices(shape, dtype=np.float64)
        y *= steps["y"]
        x *= steps["x"]
        data_dict["y"] = y.ravel()
        data_dict["x"] = x.ravel()
    return data_dict


def _get_xy_step(header: list[str]) -> dict[str, float]:
    pattern_step = re.compile(r"(?<=[XY]Step[\t\s])(.*)")
    steps = {"x": None, "y": None}
    for line in header:
        match = pattern_step.search(line)
        if match:
            step = float(match.group(0).replace(",", "."))
            if line.startswith("XStep"):
                steps["x"] = step
            elif line.startswith("YStep"):
                steps["y"] = step
    return steps


def _get_xy_cells(header: list[str]) -> dict[str, int]:
    pattern_cells = re.compile(r"(?<=[XY]Cells[\t\s])(.*)")
    cells = {"x": None, "y": None}
    for line in header:
        match = pattern_cells.search(line)
        if match:
            step = int(match.group(0))
            if line.startswith("XCells"):
                cells["x"] = step
            elif line.startswith("YCells"):
                cells["y"] = step
    return cells
