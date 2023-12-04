# -*- coding: utf-8 -*-
# Copyright 2018-2022 the orix developers
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

"""Reader of a crystal map from an Channel5 .ctf file produced by HKL
Channel5 program.
"""

from io import TextIOWrapper
from typing import List, Tuple

from diffpy.structure import Lattice, Structure
import numpy as np

from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Rotation, get_point_group

__all__ = ["file_reader"]

# MTEX has this format sorted out, check out their readers when fixing
# issues and adapting to other versions of this file format in the future:
# https://github.com/mtex-toolbox/mtex/blob/develop/interfaces/loadEBSD_ang.m
# https://github.com/mtex-toolbox/mtex/blob/develop/interfaces/loadEBSD_ACOM.m

# Plugin description
format_name = "ctf"
file_extensions = ["ctf"]
writes = False
writes_this = CrystalMap
errorCodes = {
    0: "Success",
    1: "Low band contrast",
    2: "Low band slope",
    3: "No solution",
    4: "High MAD",
    5: "Not yet analysed (job cancelled before point!)",
    6: "Unexpected error (excepts etc.)",
}


def file_reader(filename: str) -> CrystalMap:
    """Return a crystal map from HKL Channel 5 .ctf file. The crystal map
    in the input is assumed to be 2D.

    All points satisfying the following criteria are classified as not
    indexed:

    * EDAX TSL: confidence index == -1

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
    with open(filename) as f:
        header, file_data = _parse_file(f)

    # Only Grid JobMode supported
    if header["JobMode"] != "Grid":
        raise ValueError(
            "Cannot return a crystal map from the file data because only a Grid JobMode"
            "is supported"
        )

    # Get phase names and crystal symmetries from header (potentially empty)
    phase_ids, phase_names, symmetries, lattice_constants = _get_phases_from_header(
        header
    )

    structures = []
    for name, abcABG in zip(phase_names, lattice_constants):
        structures.append(Structure(title=name, lattice=Lattice(*abcABG)))

    # Get vendor properties
    props = {prop: file_data[prop] for prop in ["Bands", "Error", "MAD", "BC", "BS"]}

    # Data needed to create a CrystalMap object
    data_dict = {
        "euler1": np.radians(file_data["Euler1"]),
        "euler2": np.radians(file_data["Euler2"]),
        "euler3": np.radians(file_data["Euler3"]),
        "x": file_data["X"],
        "y": file_data["Y"],
        "phase_id": file_data["Phase"],
        "prop": props,
    }

    # Add phase list to dictionary
    data_dict["phase_list"] = PhaseList(
        names=phase_names,
        point_groups=symmetries,
        structures=structures,
        ids=phase_ids,
    )

    # Set not indexed i.e. error code is not succees
    data_dict["phase_id"][np.where(data_dict["prop"]["Error"] != 0)] = -1

    # Set scan unit
    scan_unit = "um"
    data_dict["scan_unit"] = scan_unit

    # Create rotations
    data_dict["rotations"] = Rotation.from_euler(
        np.column_stack(
            (data_dict.pop("euler1"), data_dict.pop("euler2"), data_dict.pop("euler3"))
        ),
    )

    return CrystalMap(**data_dict)


def _parse_file(file: TextIOWrapper) -> Tuple[dict, dict]:
    """Return the header and data parsed from .ctf file.

    Parameters
    ----------
    file
        File object.

    Returns
    -------
    header
        Dictionary of header parameters.
    data
        Dictionary of individual measurements
    """
    vendor = file.readline().strip()
    if vendor != "Channel Text File":
        raise ValueError("Not a Channel 5 Text File")
    header_parsed = False
    header = {}
    while not header_parsed:
        line = file.readline().split("\t", maxsplit=1)
        if line[0] == "Phases":
            n_phases = int(line[1])
            phases = []
            for i in range(n_phases):
                phase_data = file.readline().strip().split("\t")
                a, b, c = map(float, phase_data[0].split(";"))
                alpha, beta, gamma = map(float, phase_data[1].split(";"))
                try:
                    point_group = get_point_group(int(phase_data[4])).name
                except ValueError:
                    point_group = "1"
                phase = {
                    "phase_id": i + 1,
                    "name": phase_data[2],
                    "lattice_constants": [a, b, c, alpha, beta, gamma],
                    "point_group": point_group,
                    "author": phase_data[7],
                }
                phases.append(phase)
            header["Phases"] = phases
            header_parsed = True
        else:
            header[line[0]] = line[1].strip()
    # parse machine settings
    ms = header.pop("Euler angles refer to Sample Coordinate system (CS0)!").split()
    header["Settings"] = {label: value for label, value in zip(ms[0::2], ms[1::2])}
    # parse data
    data_headers = file.readline().split()
    data_array = np.fromstring(file.read(), sep="\t").reshape((-1, len(data_headers))).T
    data = {label: value for label, value in zip(data_headers, data_array)}
    # Convert selected columns to int
    for label in ["Phase", "Bands", "Error", "BC", "BS"]:
        if label in data:
            data[label] = data[label].astype(int)
    return header, data


def _get_phases_from_header(
    header: dict,
) -> Tuple[List[int], List[str], List[str], List[List[float]]]:
    """Return phase names and symmetries detected in an .ctf file
    header.

    Parameters
    ----------
    header
        Dictionary of header parameters.

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

    """
    phase_ids = []
    names = []
    point_groups = []
    lattice_constants = []
    for phase in header["Phases"]:
        phase_ids.append(phase["phase_id"])
        names.append(phase["name"])
        point_groups.append(phase["point_group"])
        lattice_constants.append(phase["lattice_constants"])

    return phase_ids, names, point_groups, lattice_constants
