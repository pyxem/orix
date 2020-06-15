# -*- coding: utf-8 -*-
# Copyright 2018-2019 The pyXem developers
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

from warnings import warn

from h5py import File
import numpy as np


# Plugin description
format_name = "orix_hdf5"
file_extensions = ["h5", "hdf5"]
format_type = "CrystalMap"
writes = True


def file_writer(filename, object2write, **kwargs):
    """Write a :class:`~orix.crystal_map.CrystalMap` object to an HDF5
    file.

    Parameters
    ----------
    filename : str
        Name of file to write to.
    object2write : CrystalMap
        Object to write to file.
    kwargs
        Keyword arguments passed to :meth:`h5py:Group.require_dataset`.
    """
    # Set manufacturer and version to use in the file
    from orix import __version__
    man_ver_dict = {"manufacturer": "orix", "version": __version__}

    # Open file in correct mode
    try:
        f = File(filename, mode="w")
    except OSError:
        raise OSError(f"Cannot write to the already open file '{filename}'.")

    # Add manufacturer and version to top group
    dict2hdf5group(man_ver_dict, f["/"], **kwargs)

    # Create scan group
    scan_group = f.create_group("scan1/crystal_map")

    # Data group with arrays with values per point
    data_group = scan_group.create_group("data")

    # Header group with all other information
    header_group = scan_group.create_group("header")
    header = _get_phase_list_dict(object2write.phases)


def dict2hdf5group(dictionary, group, **kwargs):
    """Write a dictionary to datasets in a new group in an opened HDF5
    file.

    Parameters
    ----------
    dictionary : dict
        Dataset names as keys with datasets as values.
    group : h5py:Group
        HDF5 group to write dictionary to.
    kwargs
        Keyword arguments passed to :meth:`h5py:Group.require_dataset`.
    """
    for key, val in dictionary.items():
        ddtype = type(val)
        dshape = (1,)
        if isinstance(val, dict):
            dict2hdf5group(val, group.create_group(key), **kwargs)
            continue  # Jump to next item in dictionary
        elif isinstance(val, str):
            ddtype = "S" + str(len(val) + 1)
            val = val.encode()
        elif ddtype == np.dtype("O"):
            try:
                if isinstance(val, np.ndarray):
                    ddtype = val.dtype
                else:
                    ddtype = val[0].dtype
                dshape = np.shape(val)
            except TypeError:
                warn(
                    UserWarning,
                    "The hdf5 writer could not write the following information "
                    f"to the file '{key} : {val}'."
                )
                break
        group.create_dataset(key, shape=dshape, dtype=ddtype, **kwargs)
        group[key][()] = val


def _get_phase_list_dict(phases, dictionary=None):
    """Get a dictionary of phases to write to an HDF5 group.

    Parameters
    ----------
    phases : PhaseList
        Phases to write to file.
    dictionary : dict
        Dictionary to update with phase information. If None (default),
        a new dictionary is created.

    Returns
    -------
    dictionary : dict
        Dictionary of phase information to write to the HDF5 group.
    """
    if dictionary is None:
        dictionary = {}

    dictionary["phases"] = {}
    for i, p in phases:
        phase_dict = dictionary["phases"][i]
        phase_dict["name"] = p.name
        if hasattr(p.symmetry, "name"):
            symmetry = p.symmetry.name
        else:
            symmetry = "None"
        phase_dict["symmetry"] = symmetry
        phase_dict["color"] = p.color

    return dictionary
