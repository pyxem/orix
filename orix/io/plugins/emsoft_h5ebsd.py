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

"""Reader of a crystal map from EMsoft's dictionary indexing dot product
file.
"""

import gc
import re

from diffpy.structure import Lattice, Structure
import numpy as np

from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.io.plugins._h5ebsd import H5ebsdFile
from orix.quaternion import Rotation

__all__ = ["file_reader"]

# Plugin description
format_name = "emsoft_h5ebsd"
manufacturer = "EMEBSD"
file_extensions = ["h5", "hdf5", "h5ebsd"]
writes = False
writes_this = CrystalMap


def file_reader(filename: str, refined: bool = False, **kwargs) -> CrystalMap:
    """Return a crystal map from a file in EMsoft's dictionary indexing
    dot product file format.

    Parameters
    ----------
    filename
        Path and file name.
    refined
        Whether to return refined orientations. Default is ``False``.
    **kwargs
        Keyword arguments passed to :class:`h5py.File`.

    Returns
    -------
    xmap
        Crystal map.
    """
    f = EMsoftH5ebsdFile(filename)
    f.read_refined = refined
    f.open(**kwargs)
    f.read_data_into_dictionaries()
    f.set_crystal_map_data()
    f.close()
    return f.get_crystal_map()


class EMsoftH5ebsdFile(H5ebsdFile):
    """EMsoft's HDF5 file in the h5ebsd format containing orientation
    data from dictionary indexing, to be returned as a crystal map.
    """

    dont_read_data = [
        "CIMap",
        "DictionaryEulerAngles",
        "EulerAngles",
        "Fit",
        "FZcnt",
        "IQMap",
        "ISMap",
        "Ncubochoric",
        "NumExptPatterns",
        "Phi",
        "Phi1",
        "Phi2",
        "SEM Signal",
        "Valid",
    ]
    dont_read_header = [
        "Camera Azimuthal Angle",
        "Camera Elevation Angle",
        "Coordinate System",
        "Grid Type",
        "Notes",
        "Operator",
        "Pattern Center Calibration",
        "Pattern Height",
        "Pattern Width",
        "Sample ID",
        "Sample Tilt",
        "Scan ID",
        "Working Distance",
    ]
    read_refined = False
    scan_unit = "um"

    scan_group_str = "Scan 1/"
    ebsd_group_str = scan_group_str + "EBSD/"
    data_group_str = ebsd_group_str + "Data/"
    header_group_str = ebsd_group_str + "Header/"

    def read_data_into_dictionaries(self):
        """Read data from the HDF5 file into dictionaries."""
        self.data_dict = self.get_dictionary(
            self.data_group_str, recursive=True, dont_read=self.dont_read_data
        )
        self.header_dict = self.get_dictionary(
            self.header_group_str, recursive=True, dont_read=self.dont_read_header
        )

    def set_coordinate_arrays(self):
        """Set coordinate arrays from dictionaries."""
        # Get map coordinates ("Y Position" data set is not correct in
        # EMsoft as of 2021-06, see:
        # https://github.com/EMsoft-org/EMsoft/blob/7762e1961508fe3e71d4702620764ceb98a78b9e/Source/EMsoftHDFLib/EMh5ebsd.f90#L1093)
        # self.y = self.data_dict["Y Position"][:]
        ny, nx = self.map_shape
        step_y = self.header_dict["Step Y"]
        self.y = np.sort(np.tile(np.arange(ny) * step_y, nx))
        self.x = self.data_dict["X Position"][:]

    def set_crystal_map_data(self):
        """Set necessary crystal map data from dictionaries."""
        self.set_phase_list()  # Prone to break first
        self.set_map_shape()
        self.set_coordinate_arrays()
        self.set_phase_id()
        self.set_rotations()
        self.set_properties()

    def set_map_shape(self):
        """Set the number of map rows and columns."""
        ny = self.header_dict["nRows"]
        nx = self.header_dict["nColumns"]
        self.map_shape = (ny, nx)

    def set_phase_id(self):
        """Set phase ID array from dictionaries."""
        self.phase_id = self.data_dict["Phase"][:]

    def set_phase_list(self):
        """Set phase list from dictionaries.

        This is easy, since EMsoft only outputs HDF5 files with single
        phase results.
        """
        phase = dict2phase(self.header_dict["Phase"]["1"])
        self.phase_list = PhaseList(phase)

    def set_properties(self):
        """Set dictionary of property arrays from dictionaries."""
        n_top_matches = self.file["NMLparameters/EBSDIndexingNameListType/nnk"][:][0]
        map_size = self.map_size
        expected_properties = [
            "AvDotProductMap",
            "CI",
            "IQ",
            "ISM",
            "KAM",
            "OSM",
            "RefinedDotProducts",
            "TopDotProductList",
            "TopMatchIndices",
        ]
        properties = dict()
        dd = self.data_dict
        for property_name in expected_properties:
            if property_name in dd.keys():
                prop = dd[property_name]
                if prop.shape[-1] == n_top_matches and np.prod(prop.shape) > map_size:
                    # Not a refined dot product file
                    prop = prop[:map_size].reshape(map_size, n_top_matches)
                else:
                    # Refined dot product file
                    prop = prop.reshape(map_size)
                properties[property_name] = prop
        self.properties = properties

    def set_rotations(self):
        """Set rotations from dictionaries."""
        f = self.file
        dg = f[self.data_group_str]
        if self.read_refined:
            # Radians
            euler = dg["RefinedEulerAngles"][:]
        else:  # Get n top matches for each pixel
            top_match_idx = dg["TopMatchIndices"][:][: self.map_size] - 1
            dictionary_size = dg["FZcnt"][:][0]
            # Degrees
            dictionary_euler = dg["DictionaryEulerAngles"][:][:dictionary_size]
            dictionary_euler = np.deg2rad(dictionary_euler)
            euler = dictionary_euler[top_match_idx]
        # This line is quite memory intensive
        self.rotations = Rotation.from_euler(euler)
        gc.collect()


def dict2phase(dictionary: dict) -> Phase:
    """Return a phase from a dictionary with keys and values from an
    EMsoft dot product file.

    Parameters
    ----------
    dictionary

    Returns
    -------
    phase
    """
    name = re.search(r"([A-z0-9]+)", dictionary["MaterialName"]).group(1)
    point_group = re.search(r"\[([A-z0-9/-]+)]", dictionary["Point Group"]).group(1)
    lattice = Lattice(
        *tuple(
            dictionary[f"Lattice Constant {i}"]
            for i in ["a", "b", "c", "alpha", "beta", "gamma"]
        )
    )
    structure = Structure(title=name, lattice=lattice)
    return Phase(name=name, point_group=point_group, structure=structure)
