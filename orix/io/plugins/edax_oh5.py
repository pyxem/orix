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

"""Reader of a crystal map from EDAX's (OIM) .oh5 file format."""

from diffpy.structure import Atom, Lattice, Structure
import numpy as np

from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.io.plugins._h5ebsd import H5ebsdFile
from orix.quaternion import Rotation

__all__ = ["file_reader"]

# Plugin description
format_name = "edax_oh5"
manufacturer = "EDAX"
file_extensions = ["oh5"]
writes = False
writes_this = CrystalMap


def file_reader(filename: str, **kwargs) -> CrystalMap:
    """Return a crystal map from a file in EDAX OIM's oh5
    file format.

    Parameters
    ----------
    filename
        Path and file name.
    **kwargs
        Keyword arguments passed to :class:`h5py.File`.

    Returns
    -------
    xmap
        Crystal map.
    """
    f = EDAXH5ebsdFile(filename)
    f.open(**kwargs)
    f.set_scan_group_names()
    f.set_sem_group_file_location()
    f.read_data_into_dictionaries()
    f.close()
    f.set_map_shape()  # Necessary when checking if we can read the data
    if not f.can_read():
        raise ValueError(
            "Cannot return a crystal map from the file data because only a rectangular"
            "region of interest is supported"
        )
    f.set_crystal_map_data()
    return f.get_crystal_map()


class EDAXH5ebsdFile(H5ebsdFile):
    """EDAX OIM's HDF5 file in the oh5 format containing
    orientation data from Hough indexing, to be returned as a crystal
    map.
    """

    dont_read_in_data = []
    dont_read_in_header = [
        "Camera Azimuthal Angle",
        "Camera Diameter",
        "Camera Elevation Angle",
        "Comments",
        "Coordinate System",
        "Dictionary Pattern Count",
        "File Index",
        "Hough Details",
        "Notes",
        "Pattern Center Calibration",
        "Sample Tilt",
        "Voltage[KV]",
        "Working Distance",
    ]
    dont_read_in_sem = []
    is_rectangular = True
    map_cols = None
    map_rows = None
    scan_unit = "um"

    def can_read(self) -> bool:
        """Return whether the file can be read.

        Returns
        -------
        can_read
        """
        square_grid = self.header_dict["Grid Type"] == "SqrGrid"
        return self.is_rectangular * square_grid

    def final_preparations(self):
        """Final preparations of data before creation of a crystal map."""
        if self.map_rows is not None and self.map_cols is not None:
            # Sort data points into correct order
            rc = np.array([self.map_rows, self.map_cols])
            map_order = np.ravel_multi_index(rc, self.map_shape).argsort()
            self.x = self.x[map_order]
            self.phase_id = self.phase_id[map_order]
            self.rotations = self.rotations[map_order]
            for key, value in self.properties.items():
                self.properties[key] = value[map_order]
        self.x = self.x[::-1]

    def read_data_into_dictionaries(self):
        """Read data from the HDF5 file into dictionaries."""
        if self.sem_group_location is not None:
            self.sem_dict = self.get_dictionary(
                self.sem_group_location, recursive=True, dont_read=self.dont_read_in_sem
            )
        eg_name = self.scan_groups[0] + "/EBSD/"
        self.header_dict = self.get_dictionary(
            eg_name + "Header", recursive=True, dont_read=self.dont_read_in_header
        )
        self.data_dict = self.get_dictionary(
            eg_name + "Data", recursive=True, dont_read=self.dont_read_in_data
        )

    def set_sem_group_file_location(self):
        """Set 'SEM' group HDF5 file location. This can either be
        'Scan 1/SEM' or 'Scan 1/EBSD/SEM'.
        """
        sg = self.scan_groups[0]
        potential_places = [sg, sg + "/EBSD"]
        location = None
        for pp in potential_places:
            if "SEM" in self.file[pp].keys():
                location = pp + "/SEM-PRIAS Images"
        self.sem_group_location = location

    def set_coordinate_arrays(self):
        """Set coordinate arrays from dictionaries."""
        y = self.properties["y"]
        x = self.properties["x"]
        self.y = y - np.min(y)
        self.x = x - np.min(x)

    def set_crystal_map_data(self):
        """Set necessary crystal map data from dictionaries."""
        self.set_properties()
        self.set_coordinate_arrays()
        self.set_phase_id()
        self.set_phase_list()
        self.set_rotations()
        self.final_preparations()

    def set_map_shape(self):
        """Set the number of map rows and columns. It is assumed that the
        order of the data points is correct and can be reshaped into a
        2D map without changing the order.
        """

        nrows = self.header_dict["nRows"]
        ncols = self.header_dict["nColumns"]
        self.map_shape = (nrows, ncols)

    def set_phase_id(self):
        """Set phase ID array from dictionaries."""
        self.phase_id = self.data_dict["Phase"]

    def set_phase_list(self):
        """Set phase list from dictionaries."""
        phase_list = dict2phaselist(self.header_dict["Phase"])
        phase_id = self.phase_id
        if 0 in phase_id:
            phase_list.add_not_indexed()
            phase_id[phase_id == 0] = -1
        self.phase_id = phase_id
        self.phase_list = phase_list

    def set_properties(self):
        """Set dictionary of property arrays from dictionaries."""
        self.properties = dict(
            ci=self.data_dict["CI"],
            fit=self.data_dict["Fit"],
            iq=self.data_dict["IQ"],
            y=self.data_dict["Y Position"],
            x=self.data_dict["X Position"],
        )

    def set_rotations(self):
        """Set rotations from dictionaries. EDAX saves angles in radians."""
        dd = self.data_dict
        euler = np.column_stack([dd["Phi1"], dd["Phi"], dd["Phi2"]])
        self.rotations = Rotation.from_euler(euler)


def _roi_is_rectangular(map_rows: np.ndarray, map_cols: np.ndarray) -> bool:
    """Return whether points in a map from EDAX OIM's oh5 file
    are in a rectangle.

    Parameters
    ----------
    map_rows
    map_cols

    Returns
    -------
    is_rectangular
    """
    map_rows_unique, map_rows_unique_counts = np.unique(map_rows, return_counts=True)
    map_cols_unique, map_cols_unique_counts = np.unique(map_cols, return_counts=True)
    return (
        np.all(np.diff(np.sort(map_rows_unique)) == 1)
        and np.all(np.diff(np.sort(map_cols_unique)) == 1)
        and np.unique(map_rows_unique_counts).size == 1
        and np.unique(map_cols_unique_counts).size == 1
    )


def dict2phaselist(dictionary: dict) -> PhaseList:
    """Return a list of phases from a dictionary with keys and values
    from an EDAX oh5 file.

    Parameters
    ----------
    dictionary

    Returns
    -------
    phase_list
    """
    return PhaseList(phases={int(k): dict2phase(v) for k, v in dictionary.items()})


def dict2phase(dictionary: dict) -> Phase:
    """Return a phase from a dictionary with keys and values from an EDAX oh5
    file.

    Parameters
    ----------
    dictionary

    Returns
    -------
    phase
    """
    lattice_dict = dict(
        zip(
            ["a", "b", "c", "alpha", "beta", "gamma"],
            [
                dictionary["Lattice Constant a"],
                dictionary["Lattice Constant b"],
                dictionary["Lattice Constant c"],
                dictionary["Lattice Constant alpha"],
                dictionary["Lattice Constant beta"],
                dictionary["Lattice Constant gamma"],
            ],
        )
    )
    lattice = Lattice(**lattice_dict)
    # atoms = [str2atom(atom) for atom in dictionary["AtomPositions"].values()]
    structure = Structure(lattice=lattice)  # , atoms=atoms)
    structure.title = dictionary["MaterialName"]
    laue_group = str(dictionary["LGsymID"])
    return Phase(
        name=dictionary["MaterialName"],
        point_group=_REVERSE_EDAX_POINT_GROUP_ALIASES.get(laue_group, laue_group)[0],
        structure=structure,
    )


def str2atom(atom_positions: str) -> Atom:
    """Return an atom from a string in the format used by EDAX
    in their oh5 file.

    Parameters
    ----------
    atom_positions

    Returns
    -------
    atom
    """
    atom_positions = atom_positions.split(",")
    return Atom(
        atype=atom_positions[0],
        xyz=np.array(atom_positions[1:4]),
        occupancy=int(atom_positions[-1]),
    )


# Point group alias mapping. This is needed because in EDAX TSL OIM
# Analysis 7.2, e.g. point group 432 is entered as 43.
# Used when reading a phase's point group from an EDAX OH5 file header
_REVERSE_EDAX_POINT_GROUP_ALIASES = {
    "20": ["121"],
    "2": ["2/m"],
    "22": ["222"],
    "42": ["422"],
    "32": ["321"],
    "62": ["622"],
    "43": ["432"],
    "m3m": ["m-3m"],
}
