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

"""Reader of a crystal map from an .ang file in formats produced by EDAX
TSL, NanoMegas ASTAR Index or EMsoft's EMdpmerge program.
"""

from io import TextIOWrapper
import re
from typing import List, Optional, Tuple, Union
import warnings

from diffpy.structure import Lattice, Structure
import numpy as np

from orix import __version__
from orix.crystal_map import CrystalMap, PhaseList, create_coordinate_arrays
from orix.quaternion import Rotation
from orix.quaternion.symmetry import point_group_aliases

__all__ = ["file_reader", "file_writer"]

# Plugin description
format_name = "ang"
file_extensions = ["ang"]
writes = True
writes_this = CrystalMap


def file_reader(filename: str) -> CrystalMap:
    """Return a crystal map from a file in EDAX TLS's .ang format.

    The map in the input is assumed to be 2D.

    Many vendors/programs produce an .ang file. Files from the following
    vendors/programs are tested:

    * EDAX TSL
    * NanoMegas ASTAR Index
    * EMsoft (from program `EMdpmerge`)
    * orix

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
        header = _get_header(f)

    # Phase information, potentially empty
    phases = _get_phases_from_header(header)
    phases["structures"] = []
    lattice_constants = phases.pop("lattice_constants")
    for name, abcABG in zip(phases["names"], lattice_constants):
        structure = Structure(title=name, lattice=Lattice(*abcABG))
        phases["structures"].append(structure)

    # Read all file data
    file_data = np.loadtxt(filename)

    # Get vendor and column names
    vendor, column_names = _get_vendor_columns(header, file_data.shape[1])

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
        if name in data_dict:
            data_dict[name] = file_data[:, column]
        else:
            data_dict["prop"][name] = file_data[:, column]

    # Add phase list to dictionary
    data_dict["phase_list"] = PhaseList(**phases)

    # Set which data points are not indexed
    # TODO: Add not-indexed convention for ASTAR INDEX
    if vendor in ["orix", "tsl"]:
        not_indexed = data_dict["prop"]["ci"] == -1
        data_dict["phase_id"][not_indexed] = -1

    # Set scan unit
    if vendor == "astar":
        scan_unit = "nm"
    else:
        scan_unit = "um"
    data_dict["scan_unit"] = scan_unit

    # Create rotations
    data_dict["rotations"] = Rotation.from_euler(
        np.column_stack(
            (data_dict.pop("euler1"), data_dict.pop("euler2"), data_dict.pop("euler3"))
        ),
    )

    return CrystalMap(**data_dict)


def _get_header(file: TextIOWrapper) -> List[str]:
    """Return the first lines starting with '#' in an .ang file.

    Parameters
    ----------
    file
        File object.

    Returns
    -------
    header
        List with header lines as individual elements.
    """
    header = []
    line = file.readline()
    i = 0
    # Prevent endless loop by not reading past 1 000 lines
    while line.startswith("#") and i < 1_000:
        header.append(line.rstrip())
        line = file.readline()
        i += 1
    return header


def _get_vendor_columns(header: List[str], n_cols_file: int) -> Tuple[str, List[str]]:
    """Return the .ang file column names and vendor, determined from the
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
        Determined vendor (``"tsl"``, ``"astar"``, ``"emsoft"`` or
        ``"orix"``).
    column_names
        List of column names.
    """
    # Determine vendor by searching for vendor footprint in header
    vendor_footprint = {
        "emsoft": "EMsoft",
        "astar": "ACOM",
        "orix": "Column names: phi1, Phi, phi2",
    }
    vendor = "tsl"  # Default guess
    footprint_line = None
    for name, footprint in vendor_footprint.items():
        for line in header:
            if footprint in line:
                vendor = name
                footprint_line = line
                break

    # Variants of vendor column names encountered in real data sets
    column_names = {
        "tsl": {
            0: [
                "euler1",
                "euler2",
                "euler3",
                "x",
                "y",
                "iq",  # Image quality from Hough transform
                "ci",  # Confidence index
                "phase_id",
                "detector_signal",
                "fit",  # Pattern fit
                "unknown1",
                "unknown2",
                "unknown3",
                "unknown4",
            ],
            1: [
                "euler1",
                "euler2",
                "euler3",
                "x",
                "y",
                "iq",
                "ci",
                "phase_id",
                "detector_signal",
                "fit",
            ],
        },
        "emsoft": {
            0: [
                "euler1",
                "euler2",
                "euler3",
                "x",
                "y",
                "iq",  # Image quality from Krieger Lassen's method
                "dp",  # Dot product
                "phase_id",
            ]
        },
        "astar": {
            0: [
                "euler1",
                "euler2",
                "euler3",
                "x",
                "y",
                "ind",  # Correlation index
                "rel",  # Reliability
                "phase_id",
                "relx100",  # Reliability x 100
            ],
        },
        "orix": {
            0: [
                "euler1",
                "euler2",
                "euler3",
                "x",
                "y",
                "iq",
                "ci",
                "phase_id",
                "detector_signal",
                "fit",
            ],
        },
        "unknown": {
            0: [
                "euler1",
                "euler2",
                "euler3",
                "x",
                "y",
                "unknown1",
                "unknown2",
                "phase_id",
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
            "columns: euler1, euler2, euler3, x, y, unknown1, unknown2, "
            "phase_id, unknown3, unknown4, etc."
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


def _get_phases_from_header(header: List[str]) -> dict:
    """Return phase names and symmetries detected in an .ang file
    header.

    Parameters
    ----------
    header
        List with header lines as individual elements.

    Returns
    -------
    phase_dict
        Dictionary with the following keys (and types): "ids" (int),
        "names" (str), "point_groups" (str), "lattice_constants" (list
        of floats).

    Notes
    -----
    This function has been tested with files from the following vendor's
    formats: EDAX TSL OIM Data Collection v7, ASTAR Index, and EMsoft
    v4/v5.
    """
    str_patterns = {
        "ids": "# Phase([ \t]+)([0-9 ]+)",
        "names": "# MaterialName([ \t]+)([A-z0-9 ]+)",
        "formulas": "# Formula([ \t]+)([A-z0-9 ]+)",
        "point_groups": "# Symmetry([ \t]+)([A-z0-9 ]+)",
        "lattice_constants": r"# LatticeConstants([ \t+])(.*)",
    }
    phases = {
        "ids": [],
        "names": [],
        "formulas": [],
        "point_groups": [],
        "lattice_constants": [],
    }
    for line in header:
        for key, exp in str_patterns.items():
            match = re.search(exp, line)
            if match:
                group = re.split("[ \t]", line.lstrip("# ").rstrip(" "))
                group = list(filter(None, group))
                if key == "names":
                    group = " ".join(group[1:])  # Drop "MaterialName"
                elif key == "lattice_constants":
                    group = [float(i) for i in group[1:]]
                else:
                    group = group[-1]
                phases[key].append(group)

    n_phases = len(phases["names"])

    # Use formulas in place of material names if they are all valid
    formulas = phases.pop("formulas")
    if len(formulas) == n_phases and all([len(name) for name in formulas]):
        phases["names"] = formulas

    # Ensure each phase has an ID (hopefully found in the header)
    phase_ids = [int(i) for i in phases["ids"]]
    if not len(phase_ids):
        phase_ids += [i for i in range(n_phases)]
    elif n_phases - len(phase_ids) > 0 and len(phase_ids) != 0:
        next_id = max(phase_ids) + 1
        n_left = n_phases - len(phase_ids)
        phase_ids += [i for i in range(next_id, next_id + n_left)]
    phases["ids"] = phase_ids

    return phases


def file_writer(
    filename: str,
    xmap: CrystalMap,
    index: Optional[int] = None,
    image_quality_prop: Optional[str] = None,
    confidence_index_prop: Optional[str] = None,
    detector_signal_prop: Optional[str] = None,
    pattern_fit_prop: Optional[str] = None,
    extra_prop: Union[str, List[str], None] = None,
):
    """Write a crystal map to an .ang file readable by MTEX and EDAX TSL
    OIM Analysis v7.

    The columns are phi1, Phi, phi2, x, y, image_quality,
    confidence_index, phase_id, detector_signal, and pattern_fit.

    Parameters in masked out or non-indexed points are set to:

    * euler angles = 4 * pi
    * image quality = 0
    * confidence index = -1
    * phase ID = 0 if single phase or -1 if multi phase
    * pattern fit = 180
    * detector signal = 0
    * extra properties = 0

    Parameters
    ----------
    filename
        File name with an ``".ang"`` file extension to write to.
    xmap
        Crystal map to write to file.
    index
        If the crystal map has more than one rotation/match and phase
        ID per point, this index can be set to write that "layer" of
        data to the file. For properties to be written as well, these
        must also have multiple values per point. To get the best match
        at every point, use ``None`` (default).
    image_quality_prop
        Which map property to use as the image quality. If not given
        (default), ``"iq"`` or ``"imagequality"``, if present, is used,
        otherwise just zeros. If the property has more than one value
        per point and *index* is not given, only the first value is
        used.
    confidence_index_prop
        Which map property to use as the confidence index. If not given
        (default), ``"ci"``, ``"confidenceindex"``, ``"scores"``, or
        ``"correlation"``, if present, is used, otherwise just zeros. If
        the property has more than one value per point and *index* is
        not given, only the first value is used.
    detector_signal_prop
        Which map property to use as the detector signal. If not given
        (default), ``"ds"``, or ``"detector_signal"``, if present, is
        used, otherwise just zeros. If the property has more than one
        value per point and *index* is not given, only the first value
        is used.
    pattern_fit_prop
        Which map property to use as the pattern fit. If not given
        (default), ``"fit"`` or ``"patternfit"``, if present, is used,
        otherwise just zeros. If the property has more than one value
        per point and *index* is not given, only the first value is
        used.
    extra_prop
        One or multiple properties to add as extra columns in the .ang
        file, as a string or a list of strings. If not given (default),
        no extra properties are added. If a property has more than one
        value per point and *index* is not given, only the first value
        is used.
    """
    header = _get_header_from_phases(xmap)

    # Number of decimals to round to
    decimals = 5

    # Get map data, accounting for potentially masked out values
    nrows, ncols, dy, dx = _get_nrows_ncols_step_sizes(xmap)
    map_size = nrows * ncols
    # Rotations
    if index is not None:
        eulers = xmap.get_map_data(
            xmap.rotations[:, index].to_euler(), decimals=decimals, fill_value=0
        )
    else:
        eulers = xmap.get_map_data("rotations", decimals=decimals, fill_value=0)
    eulers = eulers.reshape(map_size, 3)
    indexed_points = xmap.get_map_data(xmap.is_indexed, fill_value=False).reshape(
        map_size
    )
    eulers[~indexed_points] = 4 * np.pi

    # Coordinate arrays
    d, _ = create_coordinate_arrays((nrows, ncols), (dy, dx))
    x = d["x"]
    y = d["y"]
    x_width = _get_column_width(np.max(x))
    y_width = _get_column_width(np.max(y))

    # Properties
    desired_prop_names = [
        image_quality_prop,
        confidence_index_prop,
        detector_signal_prop,
        pattern_fit_prop,
    ]
    if extra_prop is not None:
        if isinstance(extra_prop, str):
            extra_prop = [
                extra_prop,
            ]
        desired_prop_names += extra_prop
    prop_arrays = _get_prop_arrays(
        xmap=xmap,
        prop_names=list(xmap.prop.keys()),
        desired_prop_names=desired_prop_names,
        map_size=map_size,
        index=index,
        decimals=decimals,
    )
    # Set values for non-indexed points
    prop_arrays[~indexed_points, 0::2] = 0  # IQ, detector signal
    prop_arrays[~indexed_points, 1] = -1  # CI
    prop_arrays[~indexed_points, 3] = 180  # Pattern fit
    prop_arrays[~indexed_points, 4:] = 0
    # Get property column widths
    prop_widths = [
        _get_column_width(np.max(prop_arrays[:, i]))
        for i in range(prop_arrays.shape[1])
    ]

    # Phase ID
    original_phase_ids = xmap.get_map_data("phase_id").reshape(map_size)
    pl = xmap.phases.deepcopy()
    if -1 in pl.ids:
        del pl[-1]
    if pl.size > 1:
        new_phase_ids = np.zeros(map_size, dtype=int)
    else:
        new_phase_ids = -np.ones(map_size, dtype=int)
    # Phase IDs are reversed because EDAX TSL OIM Analysis v7.2.0
    # assumes a reversed phase order in the header
    for i, (phase_id, phase) in reversed(list(enumerate(pl))):
        new_phase_ids[original_phase_ids == phase_id] = i + 1

    # Extend header with column names
    header += (
        "\n"
        "Column names: phi1, Phi, phi2, x, y, image_quality, confidence_index, "
        "phase_id, detector_signal, pattern_fit"
    )

    # Get data formats
    fmt = (
        f"%8.5f  %8.5f  %8.5f  %{x_width}.5f  %{y_width}.5f  %{prop_widths[0]}.5f  "
        f"%{prop_widths[1]}.5f  %2i  %{prop_widths[2]}.5f  %{prop_widths[3]}.5f"
    )
    if extra_prop is not None:  # Then it must be a list!
        for extra_width, extra_prop_name in zip(prop_widths[4:], extra_prop):
            fmt += f"  %{extra_width}.5f"
            header += f", {extra_prop_name}"
    header += "\n"

    # Finally, write everything to file
    np.savetxt(
        fname=filename,
        X=np.column_stack(
            [eulers, x, y, prop_arrays[:, :2], new_phase_ids, prop_arrays[:, 2:]]
        ),
        fmt=fmt,
        header=header,
    )


def _get_header_from_phases(xmap: CrystalMap) -> str:
    """Return a string with the .ang file header from the crystal map
    metadata.

    Parameters
    ----------
    xmap
        Crystal map.

    Returns
    -------
    header
        File header.

    Notes
    -----
    Phase IDs are reversed because EDAX TSL OIM Analysis v7.2.0 assumes
    a reversed phase order in the header
    """
    nrows, ncols, _, _ = _get_nrows_ncols_step_sizes(xmap)
    # Initialize header with microscope info
    header = (
        "TEM_PIXperUM           1.000000\n"
        "x-star                 0.000000\n"
        "y-star                 0.000000\n"
        "z-star                 0.000000\n"
        "WorkingDistance        0.000000\n"
        "\n"
    )
    # Get phase list, removing the non indexed phase if present
    pl = xmap.phases.deepcopy()
    if -1 in pl.ids:
        del pl[-1]

    # Extend header with phase info
    # Phase IDs are reversed because EDAX TSL OIM Analysis v7.2.0
    # assumes a reversed phase order in the header
    for i, (_, phase) in reversed(list(enumerate(pl))):
        lattice_constants = phase.structure.lattice.abcABG()
        lattice_constants = " ".join([f"{float(val):.3f}" for val in lattice_constants])
        phase_id = i + 1
        phase_name = phase.name
        if phase_name == "":
            phase_name = f"phase{phase_id}"
        if not phase.point_group:
            point_group_name = "1"
        else:
            proper_point_group = phase.point_group.proper_subgroup
            point_group_name = proper_point_group.name
            for key, alias in point_group_aliases.items():
                if point_group_name == key:
                    point_group_name = alias[0]
                    break
        header += (
            f"Phase {phase_id}\n"
            f"MaterialName    {phase_name}\n"
            f"Formula    {phase_name}\n"
            f"Info\n"
            f"Symmetry    {point_group_name}\n"
            f"LatticeConstants    {lattice_constants}\n"
            "NumberFamilies    0\n"
        )
    # Extend header with map info
    header += (
        "GRID: SqrGrid\n"
        f"XSTEP: {float(xmap.dx):.6f}\n"
        f"YSTEP: {float(xmap.dy):.6f}\n"
        f"NCOLS_ODD: {ncols}\n"
        f"NCOLS_EVEN: {ncols}\n"
        f"NROWS: {nrows}\n"
        "\n"
        f"OPERATOR: orix_v{__version__}\n"
        "\n"
        "SAMPLEID:\n"
        "\n"
        "SCANID:\n"
    )
    return header


def _get_nrows_ncols_step_sizes(xmap: CrystalMap) -> Tuple[int, int, float, float]:
    """Get crystal map shape and step sizes.

    Parameters
    ----------
    xmap
        Crystal map.

    Returns
    -------
    nrows
    ncols
    dy
    dx
    """
    nrows = ncols = 1
    dy, dx = xmap.dy, xmap.dx
    if xmap.ndim == 1:
        ncols = xmap.shape[0]
        dy = 1
    else:  # xmap.ndim == 2:
        nrows, ncols = xmap.shape
    return nrows, ncols, dy, dx


def _get_column_width(max_value: int, decimals: int = 5) -> int:
    """Get width of column to pass to :func:`numpy.savetxt`, accounting
    for the decimal point and a sign +/-.

    Parameters
    ----------
    max_value
    decimals

    Returns
    -------
    column_width
    """
    return len(str(int(max_value // 1))) + decimals + 2


def _get_prop_arrays(
    xmap: CrystalMap,
    prop_names: List[str],
    desired_prop_names: List[str],
    map_size: int,
    index: Optional[int],
    decimals: int = 5,
) -> np.ndarray:
    """Return a 2D array (n_points, n_properties) with desired property
    values in, or just zeros.

    This function tries to get as many properties as possible from the
    crystal map properties.

    Parameters
    ----------
    xmap
    prop_names
    desired_prop_names
    map_size
    index
    decimals

    Returns
    -------
    prop_arrays
    """
    # "Image_quality" -> "imagequality" etc.
    prop_names_lower = [k.lower().replace("_", "") for k in prop_names]
    prop_names_lower_arr = np.array(prop_names_lower)
    # Potential extra names added so that lists are of the same length
    # in the loop
    all_expected_prop_names = [
        ["iq", "imagequality"],
        ["ci", "confidenceindex", "scores", "correlation"],
        ["ss", "semsignal", "detectorsignal"],
        ["fit", "patternfit"],
    ] + desired_prop_names[4:]
    n_desired_props = len(desired_prop_names)
    prop_arrays = np.zeros((map_size, n_desired_props), dtype=np.float32)
    for i, (name, names) in enumerate(zip(desired_prop_names, all_expected_prop_names)):
        prop = _get_prop_array(
            xmap=xmap,
            prop_name=name,
            expected_prop_names=names,
            prop_names=prop_names,
            prop_names_lower_arr=prop_names_lower_arr,
            decimals=decimals,
            index=index,
        )
        if prop is not None:
            prop_arrays[:, i] = prop.reshape(map_size)
    return prop_arrays


def _get_prop_array(
    xmap: CrystalMap,
    prop_name: str,
    expected_prop_names: List[str],
    prop_names: List[str],
    prop_names_lower_arr: np.ndarray,
    index: Optional[int],
    decimals: int = 5,
    fill_value: Union[int, float, bool] = 0,
) -> Union[np.ndarray, None]:
    """Return a 1D array (n_points,) with the desired property values or
    ``None`` if the property cannot be read.

    Reasons for why the property cannot be read:

    * Property name isn't among the crystal map properties
    * Property has only one value per point, but *index* is not ``None``

    Parameters
    ----------
    xmap
    prop_name
    expected_prop_names
    prop_names
    prop_names_lower_arr
    index
    decimals
    fill_value

    Returns
    -------
    prop_array
        Property array or none if none found.
    """
    kwargs = dict(decimals=decimals, fill_value=fill_value)
    if not len(prop_names_lower_arr) and not prop_name:
        return
    else:
        if not prop_name:
            # Search for a suitable property
            for k in expected_prop_names:
                is_equal = k == prop_names_lower_arr
                if is_equal.any():
                    prop_name = prop_names[np.argmax(is_equal)]
                    break
            else:  # If no suitable property was found
                return
        # There is a property
        if len(xmap.prop[prop_name].shape) == 1:
            # Return the single array even if `index` is given
            return xmap.get_map_data(prop_name, **kwargs)
        else:
            if not index:
                index = 0
            return xmap.get_map_data(xmap.prop[prop_name][:, index], **kwargs)
