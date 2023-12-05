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

"""Reader of a crystal map from an .ctf file in formats produced by
Oxford AZtec and EMsoft's EMdpmerge program.
"""

from io import TextIOWrapper
from typing import List, Tuple
import warnings

from diffpy.structure import Lattice, Structure
import numpy as np

from orix import __version__
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Rotation

__all__ = ["file_reader"]

# Plugin description
format_name = "ctf"
file_extensions = ["ctf"]
writes = False
writes_this = None


def file_reader(filename: str) -> CrystalMap:
    """Return a crystal map from a file in Oxford AZtec HKL's .ctf format. The
    map in the input is assumed to be 2D.

    Many vendors produce an .ctf file. Supported vendors are:

    * Oxford AZtec HKL
    * EMsoft (from program `EMdpmerge`)


    All points with a phase of 0 are classified as not indexed.

    Parameters
    ----------
    filename
        Path and file name.

    Returns
    -------
    xmap
        Crystal map.
    """
    # Get file header
    with open(filename, "r") as f:
        [header, data_starting_row] = _get_header(f)

    # Get phase names and crystal symmetries from header (potentially empty)
    phase_ids, phase_names, symmetries, lattice_constants = _get_phases_from_header(
        header
    )
    structures = []
    for name, abcABG in zip(phase_names, lattice_constants):
        structures.append(Structure(title=name, lattice=Lattice(*abcABG)))

    # Read all file data
    file_data = np.loadtxt(filename, skiprows=data_starting_row)

    # Get vendor and column names
    n_rows, n_cols = file_data.shape

    column_names = (
        [
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
            "BS",  # Band Slope
        ],
    )

    # Data needed to create a CrystalMap object
    data_dict = {
        "euler1": None,
        "euler2": None,
        "euler3": None,
        "x": None,
        "y": None,
        "phase_id": None,
        "prop": {},
    }
    for column, name in enumerate(column_names):
        if name in data_dict.keys():
            data_dict[name] = file_data[:, column]
        else:
            data_dict["prop"][name] = file_data[:, column]

    # Add phase list to dictionary
    data_dict["phase_list"] = PhaseList(
        names=phase_names,
        space_groups=symmetries,
        structures=structures,
        ids=phase_ids,
    )

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


def _get_header(file: TextIOWrapper) -> List[str]:
    """Return the first lines above the mapping data and the data starting row number
    in an .ctf file.

    Parameters
    ----------
    file
        File object.

    Returns
    -------
    header
        List with header lines as individual elements.
    data_starting_row
        The starting row number for the data lines
    """
    header = []
    line = file.readline()
    i = 0
    while not line.startswith("Phase\tX\tY"):
        header.append(line.rstrip())
        i += 1
        line = file.readline()
    return header, i + 1


def _get_phases_from_header(
    header: List[str],
) -> Tuple[List[int], List[str], List[str], List[List[float]]]:
    """Return phase names and symmetries detected in an .ctf file
    header.

    Parameters
    ----------
    header
        List with header lines as individual elements.

    Returns
    -------
    ids
        Phase IDs.
    phase_names
        List of names of detected phases.
    phase_point_groups
        List of point groups of detected phase.
    lattice_constants
        List of list of lattice parameters of detected phases.

    Notes
    -----
    Regular expressions are used to collect phase name, formula and
    point group. This function have been tested with files from the
    following vendor's formats: Oxford AZtec HKL v5/v6, and EMsoft v4/v5.
    """
    phases = {
        "name": [],
        "space_group": [],
        "lattice_constants": [],
        "id": [],
    }
    for i, line in enumerate(header):
        if line.startswith("Phases"):
            break

    n_phases = int(line.split("\t")[1])

    for j in range(n_phases):
        phase_data = header[i + 1 + j].split("\t")
        phases["name"].append(phase_data[2])
        phases["space_group"].append(int(phase_data[4]))
        phases["lattice_constants"].append(
            [float(i) for i in phase_data[0].split(";") + phase_data[1].split(";")]
        )
        phases["id"].append(j + 1)

    names = phases["name"]
    phase_ids = [int(i) for i in phases["id"]]

    return phase_ids, names, phases["space_group"], phases["lattice_constants"]
