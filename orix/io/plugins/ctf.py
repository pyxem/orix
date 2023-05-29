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

# from orix import __version__
from orix.crystal_map import CrystalMap, PhaseList
from orix.quaternion import Rotation

__all__ = ["file_reader"]


def file_reader(filename: str) -> CrystalMap:
    """Return a crystal map from a file in Oxford AZtec HKL's .ctf format. The
    map in the input is assumed to be 2D.

    Many vendors produce an .ctf file. Supported vendors are:

    * Oxford AZtec HKL
    * EMsoft (from program `EMdpmerge`)
    * orix

    All points satisfying the following criteria are classified as not
    indexed:

    * Oxford AZtec HKL: Phase == 0

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
    vendor, column_names = _get_vendor_columns(header, n_cols)

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
    scan_unit = "um"
    data_dict["scan_unit"] = scan_unit

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
    all_data = [line.rstrip() for line in file.readlines()]

    phase_num_row = 0
    phases_num_line = str()
    for line in all_data:
        if "Phases" in line:
            phases_num_line = line
            break
        phase_num_row += 1
    if phases_num_line:
        try:
            phase_num = int(phases_num_line.split("\t")[1])
            header = all_data[: (phase_num_row + phase_num + 1)]
            data_starting_row = phase_num_row + phase_num + 2
        except:
            header = None
            data_starting_row = None
            warnings.warn(
                f"Total number of phases has to be defined in the .ctf file."
                f"No such information can be found. Incompatible file format."
            )
    else:
        header = None
        data_starting_row = None
        warnings.warn(
            f"Total number of phases has to be defined in the .ctf file."
            f"No such information can be found. Incompatible file format."
        )

    return header, data_starting_row


def _get_vendor_columns(header: List[str], n_cols_file: int) -> Tuple[str, List[str]]:
    """Return the .ctf file column names and vendor, determined from the
    header.

    Parameters
    ----------
    header
        List with header lines as individual elements.
    n_cols_file
        Number of file columns.

    Returns
    -------
    vendor
        Determined vendor (``"hkl"``, ``"emsoft"`` or ``"orix"``).
    column_names
        List of column names.
    """
    # Assume Oxford TSL by default
    vendor = "hkl"

    # Determine vendor by searching for the vendor footprint in the header
    vendor_footprint = {
        "emsoft": "EMsoft",
        "orix": "Column names: phi1, Phi, phi2",
    }
    footprint_line = None
    for name, footprint in vendor_footprint.items():
        for line in header:
            if footprint in line:
                vendor = name
                footprint_line = line
                break

    # Variants of vendor column names encountered in real data sets
    column_names = {
        "hkl": {
            0: [
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
        },
        "emsoft": {
            0: [
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
            ]
        },
        "orix": {
            0: [
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
        },
        "unknown": {
            0: [
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
            ]
        },
    }

    n_variants = len(column_names[vendor])
    n_cols_expected = [len(column_names[vendor][k]) for k in range(n_variants)]
    if vendor == "orix" and "Column names" in footprint_line:
        # Append names of extra properties found, if any, in the orix
        # .ang file header
        vendor_column_names = column_names[vendor][0]
        n_cols = n_cols_expected[0]
        extra_props = footprint_line.split(":")[1].split(",")[n_cols:]
        vendor_column_names += [i.lstrip(" ").replace(" ", "_") for i in extra_props]
    elif n_cols_file not in n_cols_expected:
        warnings.warn(
            f"Number of columns, {n_cols_file}, in the file is not equal to "
            f"the expected number of columns, {n_cols_expected}, for the \n"
            f"assumed vendor '{vendor}'. Will therefore assume the following "
            "columns: phase_id, x, y, bands, error, euler1, euler2, euler3"
            "MAD, BC, BS, etc."
        )
        vendor = "unknown"
        vendor_column_names = column_names[vendor][0]
        n_cols = len(vendor_column_names)
        if n_cols_file > n_cols:
            # Add any extra columns as properties
            for i in range(n_cols_file - n_cols):
                vendor_column_names.append("unknown" + str(i + 3))
    else:
        idx = np.where(np.equal(n_cols_file, n_cols_expected))[0][0]
        vendor_column_names = column_names[vendor][idx]

    return vendor, vendor_column_names


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
