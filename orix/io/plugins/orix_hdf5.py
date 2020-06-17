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
module = "orix.crystal_map"
format_type = "CrystalMap"
writes = True


def file_writer(filename, crystal_map, **kwargs):
    """Write a :class:`~orix.crystal_map.CrystalMap` object to an HDF5
    file.

    Parameters
    ----------
    filename : str
        Name of file to write to.
    crystal_map : CrystalMap
        Object to write to file.
    kwargs
        Keyword arguments passed to :meth:`h5py:Group.require_dataset`.
    """
    # Open file in correct mode
    try:
        f = File(filename, mode="w")
    except OSError:
        raise OSError(f"Cannot write to the already open file '{filename}'.")

    from orix import __version__

    eulers = crystal_map._rotations.to_euler()
    file_dict = {
        "manufacturer": "orix",
        "version": __version__,
        "crystal_map": {
            "data": {
                "z": crystal_map._z,
                "y": crystal_map._y,
                "x": crystal_map._x,
                "phi1": eulers[:, 0],
                "Phi": eulers[:, 1],
                "phi2": eulers[:, 2],
                "phase_id": crystal_map._phase_id,
                "id": crystal_map._id,
                "is_in_data": crystal_map.is_in_data,
            },
            "header": {
                "grid_type": "square",
                "nz": len(crystal_map._z) if hasattr(crystal_map._z, "__iter__") else 1,
                "ny": len(crystal_map._y) if hasattr(crystal_map._y, "__iter__") else 1,
                "nx": len(crystal_map._x) if hasattr(crystal_map._x, "__iter__") else 1,
                "z_step": crystal_map.dz,
                "y_step": crystal_map.dy,
                "x_step": crystal_map.dx,
                "rotations_per_point": crystal_map.rotations_per_point,
                "scan_unit": crystal_map.scan_unit,
            }
        }
    }
    dict2hdf5group(file_dict, f["/"], **kwargs)

    # Header group with all other information
#    header_group = scan_group.create_group("header")
#    header = _get_phase_list_dict(crystal_map.phases)

    f.close()


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
