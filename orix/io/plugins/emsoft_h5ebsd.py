# -*- coding: utf-8 -*-
# Copyright 2018-2020 The pyXem developers
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

import re

import h5py
import numpy as np

from orix.quaternion.rotation import Rotation

# Plugin description
format_name = "emsoft_h5ebsd"
file_extensions = ["h5", "hdf5", "h5ebsd"]
module = "orix.crystal_map"
format_type = "CrystalMap"
writes = False


def file_reader(filename, refined=False, **kwargs):
    """Return a CrystalMap object from EMsoft's dictionary indexing dot
    product files.

    Parameters
    ----------
    filename : str
        Path and file name.
    refined : bool, optional
        Whether to return refined orientations (default is ``False``).
    kwargs
        Keyword arguments passed to :func:`h5py.File`.

    Returns
    -------
    dict
    """
    mode = kwargs.pop("mode", "r")
    f = h5py.File(filename, mode=mode, **kwargs)

    # Get groups for convenience
    ebsd_group = f["Scan 1/EBSD"]
    data_group = ebsd_group["Data"]
    header_group = ebsd_group["Header"]
    phase_group = header_group["Phase/1"]

    # Get map shape and step sizes
    ny = header_group["nRows"][:][0]
    nx = header_group["nColumns"][:][0]
    step_y = header_group["Step Y"][:][0]
    map_size = ny * nx

    # Some of the data needed to create a CrystalMap object
    data_dict = {
        # Get map coordinates ("Y Position" data set is not correct in EMsoft as of
        # 2020-04, see:
        # https://github.com/EMsoft-org/EMsoft/blob/7762e1961508fe3e71d4702620764ceb98a78b9e/Source/EMsoftHDFLib/EMh5ebsd.f90#L1093)
        "x": data_group["X Position"][:],
        # y = data_group["Y Position"][:]
        "y": np.sort(np.tile(np.arange(ny) * step_y, nx)),
        # Get phase IDs
        "phase_id": data_group["Phase"][:],
        # Get phase name and crystal symmetry
        "phase_name": re.search(
            r"([A-z0-9]+)", phase_group["MaterialName"][:][0].decode()
        ).group(1),
        "symmetry": re.search(
            r"\[([A-z0-9]+)\]", phase_group["Point Group"][:][0].decode()
        ).group(1)
    }

    # Get rotations
    if refined:
        euler = data_group["RefinedEulerAngles"][:]
    else:  # Get n top matches for each pixel
        top_match_idx = data_group["TopMatchIndices"][:][:map_size]
        dictionary_size = data_group["FZcnt"][:][0]
        dictionary_euler = data_group["DictionaryEulerAngles"][:][:dictionary_size]
        euler = dictionary_euler[top_match_idx, :]
    data_dict["rotations"] = Rotation.from_euler(euler)

    # Get number of top matches kept per data point
    n_top_matches = f["NMLparameters/EBSDIndexingNameListType/nnk"][:][0]

    data_dict["prop"] = _get_properties(
        data_group=data_group, n_top_matches=n_top_matches, map_size=map_size,
    )

    return data_dict


def _get_properties(data_group, n_top_matches, map_size):
    """Return a dictionary of properties within an EMsoft h5ebsd file, with
    property names as the dictionary key and arrays as the values.

    Parameters
    ----------
    data_group : h5py.Group
        HDF5 group with the property data sets.
    n_top_matches : int
        Number of rotations per point.
    map_size : int
        Data size.

    Returns
    -------
    properties : dict
        Property dictionary.
    """
    expected_properties = [
        "AvDotProductMap",
        "CI",
        "CIMap",
        "IQ",
        "IQMap",
        "ISM",
        "ISMap",
        "KAM",
        "OSM",
        "RefinedDotProducts",
        "TopDotProductList",
        "TopMatchIndices",
    ]

    # Get properties
    properties = {}
    for property_name in expected_properties:
        if property_name in data_group.keys():
            prop = data_group[property_name][:]
            if prop.shape[-1] == n_top_matches:
                prop = prop[:map_size].reshape((map_size,) + (n_top_matches,))
            else:
                prop = prop.reshape(map_size)
            properties[property_name] = prop

    return properties
