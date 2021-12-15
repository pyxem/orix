# -*- coding: utf-8 -*-
# Copyright 2018-2021 the orix developers
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

"""Reader of a crystal map from Bruker's h5ebsd file format."""

from diffpy.structure import Atom, Lattice, Structure
import numpy as np

from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.io.plugins._h5ebsd import H5ebsdFile
from orix.quaternion import Rotation


__all__ = ["file_reader"]

# Plugin description
format_name = "bruker_h5ebsd"
manufacturer = "Bruker"
file_extensions = ["h5", "hdf5", "h5ebsd"]
writes = False
writes_this = CrystalMap


def file_reader(filename, **kwargs):
    """Return a :class:`~orix.crystal_map.CrystalMap` from a file in
    Bruker Nano's dot product file format.

    Parameters
    ----------
    filename : str
        Path and file name.
    kwargs
        Keyword arguments passed to :func:`h5py.File`.

    Returns
    -------
    CrystalMap
    """
    f = BrukerH5ebsdFile(filename)
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


class BrukerH5ebsdFile(H5ebsdFile):
    """Bruker Nano's HDF5 file in the h5ebsd format containing
    orientation data from Hough indexing, to be returned as a crystal
    map.
    """

    dont_read_in_data = ["RawPatterns"]
    dont_read_in_header = [
        "CameraTilt",
        "Coordinate Systems",
        "DetectorFullHeightMicrons",
        "DetectorFullWidthMicrons",
        "KV",
        "MADMax",
        "Magnification",
        "MapStepFactor",
        "MaxRadonBandCount",
        "MinIndexedBands",
        "NPoints",
        "OriginalFile",
        "PatternHeight",
        "PatternWidth",
        "PixelByteCount",
        "SEPixelSizeX",
        "SEPixelSizeY",
        "SampleTilt",
        "TopClip",
        "UnClippedPatternHeight",
        "WD",
        "XSTEP",
        "YSTE",
        "ZOffset",
    ]
    dont_read_in_sem = ["ZOffset"]
    is_rectangular = True
    map_cols = None
    map_rows = None
    scan_unit = "um"

    def can_read(self):
        """Return whether the file can be read."""
        square_grid = self.header_dict["Grid Type"] == "isometric"
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
                location = pp + "/SEM"
        self.sem_group_location = location

    def set_coordinate_arrays(self):
        """Set coordinate arrays from dictionaries."""
        y = self.properties["YSAMPLE"]
        x = self.properties["XSAMPLE"]
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
        """Set the number of map rows and columns.

        Also try to set the map row and column position of each point in
        the data arrays, if the data sets 'Scan 1/SEM/IY' and
        'Scan 1/SEM/IX' are present. If not, it is assumed that the
        order of the data points is correct, and can be reshaped into a
        2D map without changing the order.
        """
        sd = self.sem_dict
        potential_names_y = ["IY", "SEM IY"]
        potential_names_x = ["IX", "SEM IX"]
        match_y = None
        match_x = None
        for key in sd.keys():
            if key in potential_names_y:
                match_y = key
            elif key in potential_names_x:
                match_x = key
        if match_y is not None and match_x is not None:
            map_rows = self.sem_dict[match_y]
            map_cols = self.sem_dict[match_x]

            # If False, we cannot read the data
            self.is_rectangular = _roi_is_rectangular(map_rows, map_cols)

            min_r, max_r = np.min(map_rows), np.max(map_rows)
            min_c, max_c = np.min(map_cols), np.max(map_cols)
            nrows = max_r - min_r + 1
            ncols = max_c - min_c + 1
            self.map_rows = map_rows - min_r
            self.map_cols = map_cols - min_c
        else:
            nrows = self.header_dict["NROWS"]
            ncols = self.header_dict["NCOLS"]
        self.map_shape = (nrows, ncols)

    def set_phase_id(self):
        """Set phase ID array from dictionaries."""
        self.phase_id = self.data_dict["Phase"]

    def set_phase_list(self):
        """Set phase list from dictionaries."""
        phase_list = dict2phaselist(self.header_dict["Phases"])
        phase_id = self.phase_id
        if 0 in phase_id:
            phase_list.add_not_indexed()
            phase_id[phase_id == 0] = -1
        self.phase_id = phase_id
        self.phase_list = phase_list

    def set_properties(self):
        """Set dictionary of property arrays from dictionaries."""
        self.properties = dict(
            PCX=self.data_dict["PCX"],
            PCY=self.data_dict["PCY"],
            DD=self.data_dict["DD"],
            MAD=self.data_dict["MAD"],
            MADPhase=self.data_dict["MADPhase"],
            NIndexedBands=self.data_dict["NIndexedBands"],
            RadonBandCount=self.data_dict["RadonBandCount"],
            RadonQuality=self.data_dict["RadonQuality"],
            XBEAM=self.data_dict["X BEAM"],
            YBEAM=self.data_dict["Y BEAM"],
            XSAMPLE=self.data_dict["X SAMPLE"],
            YSAMPLE=self.data_dict["Y SAMPLE"],
            ZSAMPLE=self.data_dict["Z SAMPLE"],
        )

    def set_rotations(self):
        """Set rotations from dictionaries."""
        dd = self.data_dict
        euler = np.column_stack([dd["phi1"], dd["PHI"], dd["phi2"]])
        euler = np.deg2rad(euler)
        self.rotations = Rotation.from_euler(euler)


def _roi_is_rectangular(map_rows, map_cols):
    """Return whether points in a map from Bruker Nano's h5ebsd file
    are in a rectangle.

    Parameters
    ----------
    map_rows : numpy.ndarray
    map_cols : numpy.ndarray

    Returns
    -------
    bool
    """
    map_rows_unique, map_rows_unique_counts = np.unique(map_rows, return_counts=True)
    map_cols_unique, map_cols_unique_counts = np.unique(map_cols, return_counts=True)
    return (
        np.all(np.diff(np.sort(map_rows_unique)) == 1)
        and np.all(np.diff(np.sort(map_cols_unique)) == 1)
        and np.unique(map_rows_unique_counts).size == 1
        and np.unique(map_cols_unique_counts).size == 1
    )


def dict2phaselist(dictionary):
    """Return a list of phases from a dictionary with keys and values
    from a Bruker Nano h5ebsd file.

    Parameters
    ----------
    dictionary : dict

    Returns
    -------
    PhaseList
    """
    return PhaseList(phases={int(k): dict2phase(v) for k, v in dictionary.items()})


def dict2phase(dictionary):
    """Return a phase from a dictionary with keys and values from a
    Bruker Nano h5ebsd file.

    Parameters
    ----------
    dictionary : dict

    Returns
    -------
    Phase
    """
    lattice_dict = dict(
        zip(["a", "b", "c", "alpha", "beta", "gamma"], dictionary["LatticeConstants"])
    )
    lattice = Lattice(**lattice_dict)
    atoms = [str2atom(atom) for atom in dictionary["AtomPositions"].values()]
    structure = Structure(lattice=lattice, atoms=atoms)
    structure.title = dictionary["Name"]
    return Phase(
        name=dictionary["Name"], space_group=int(dictionary["IT"]), structure=structure
    )


def str2atom(atom_positions):
    """Return an atom from a string in the format used by Bruker Nano
    in their h5ebsd file.

    Parameters
    ----------
    atom_positions : str

    Returns
    -------
    diffpy.structure.Atom
    """
    atom_positions = atom_positions.split(",")
    return Atom(
        atype=atom_positions[0],
        xyz=np.array(atom_positions[1:4]),
        occupancy=int(atom_positions[-1]),
    )
