# -*- coding: utf-8 -*-
# Copyright 2018-2020 the orix developers
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

import copy
from warnings import warn

from diffpy.structure import Atom, Lattice, Structure
from h5py import File, Group
import numpy as np

from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.quaternion.rotation import Rotation

# Plugin description
format_name = "orix_hdf5"
file_extensions = ["h5", "hdf5"]
writes = True
writes_this = CrystalMap
footprint = ["crystal_map"]  # Unique HDF5 footprint
# TODO: Extend reader/writer to Phase and PhaseList objects


def file_reader(filename, **kwargs):
    """Return a :class:`~orix.crystal_map.crystal_map.CrystalMap` object
    from a file in orix' HDF5 file format.

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
    mode = kwargs.pop("mode", "r")
    with File(filename, mode=mode, **kwargs) as f:
        file_dict = hdf5group2dict(f["/"], recursive=True)
    return dict2crystalmap(file_dict["crystal_map"])


def hdf5group2dict(group, dictionary=None, recursive=False):
    """Return a dictionary with values from datasets in a group in an
    opened HDF5 file.

    Parameters
    ----------
    group : h5py:Group
        HDF5 group object.
    dictionary : dict, optional
        To fill dataset values into. If None (default), a new dictionary
        is created.
    recursive : bool, optional
        Whether to add subgroups to dictionary. Default is False.

    Returns
    -------
    dictionary : dict
        Dataset values in group (and subgroups if recursive=True).
    """
    if dictionary is None:
        dictionary = {}
    for key, val in group.items():
        # Check whether to extract subgroup or write value the dictionary
        if isinstance(val, Group):
            if recursive:
                dictionary[key] = {}
                hdf5group2dict(
                    group=group[key], dictionary=dictionary[key], recursive=recursive,
                )
            else:
                dictionary[key] = val
        else:
            val = val[()]
            # Prepare value for entry in dictionary
            if isinstance(val, np.ndarray) and len(val) == 1:
                val = val[0]
            if isinstance(val, bytes):
                val = val.decode("latin-1")
            dictionary[key] = val
    return dictionary


def dict2crystalmap(dictionary):
    """Get a crystal map from necessary items in a dictionary.

    Parameters
    ----------
    dictionary : dict
        Dictionary with crystal map information.

    Returns
    -------
    CrystalMap
    """
    dictionary = copy.deepcopy(dictionary)

    data = dictionary["data"]
    header = dictionary["header"]

    # New dictionary with CrystalMap initialization arguments as keys
    crystal_map_dict = {
        # Use dstack and squeeze to allow more rotations per data point
        "rotations": Rotation.from_euler(
            np.dstack((data.pop("phi1"), data.pop("Phi"), data.pop("phi2"))).squeeze(),
        ),
        "scan_unit": header["scan_unit"],
        "phase_list": dict2phaselist(header["phases"]),
        "phase_id": data.pop("phase_id"),
        "is_in_data": data.pop("is_in_data"),
    }
    # Add standard items by updating the new dictionary
    for direction in ["z", "y", "x"]:
        this_direction = data.pop(direction)
        if hasattr(this_direction, "__iter__") is False:
            this_direction = None
        crystal_map_dict[direction] = this_direction
    _ = [data.pop(i) for i in ["id"]]
    # What's left should be properties like quality metrics etc.
    crystal_map_dict.update({"prop": data})

    return CrystalMap(**crystal_map_dict)


def dict2phaselist(dictionary):
    """Get a :class:`~orix.crystal_map.phase_list.PhaseList` object from a
    dictionary.

    Parameters
    ----------
    dictionary : dict
        Dictionary with phase list information.

    Returns
    -------
    PhaseList
    """
    dictionary = copy.deepcopy(dictionary)
    return PhaseList(phases={int(k): dict2phase(v) for k, v in dictionary.items()})


def dict2phase(dictionary):
    """Get a :class:`~orix.crystal_map.phase_list.Phase` object from a
    dictionary.

    Parameters
    ----------
    dictionary : dict
        Dictionary with phase information.

    Returns
    -------
    Phase
    """
    dictionary = copy.deepcopy(dictionary)
    structure = dict2structure(dictionary["structure"])
    structure.title = dictionary["name"]
    # TODO: Remove this check in v0.6.0, since space_group was introduced in v0.4.0
    try:
        space_group = dictionary["space_group"]  # Either "None" or int
    except KeyError:  # v0.3.0
        space_group = "None"
    if space_group == "None":
        space_group = None
    else:
        space_group = int(space_group)
    # TODO: Remove this check in v0.6.0, since name change was introduced in v0.4.0
    try:
        point_group = dictionary["point_group"]
    except KeyError:  # v0.3.0
        point_group = dictionary["symmetry"]
    if point_group == "None":
        point_group = None
    return Phase(
        name=dictionary["name"],
        space_group=space_group,
        point_group=point_group,
        structure=structure,
        color=dictionary["color"],
    )


def dict2structure(dictionary):
    """Get a :class:`diffpy.structure.Structure` object from a dictionary.

    Parameters
    ----------
    dictionary : dict
        Dictionary with structure information.

    Returns
    -------
    Structure
    """
    dictionary = copy.deepcopy(dictionary)
    return Structure(
        lattice=dict2lattice(dictionary["lattice"]),
        atoms=[dict2atom(atom) for atom in dictionary["atoms"].values()],
    )


def dict2lattice(dictionary):
    """Get a :class:`diffpy.structure.Lattice` object from a dictionary.

    Parameters
    ----------
    dictionary : dict
        Dictionary with lattice information.

    Returns
    -------
    Lattice
    """
    dictionary = copy.deepcopy(dictionary)
    lattice_dict = {
        k: v
        for k, v in zip(["a", "b", "c", "alpha", "beta", "gamma"], dictionary["abcABG"])
    }
    lattice_dict["baserot"] = dictionary["baserot"]
    return Lattice(**lattice_dict)


def dict2atom(dictionary):
    """Get a :class:`diffpy.structure.Atom.atom` object from a dictionary.

    Parameters
    ----------
    dictionary : dict
        Dictionary with atom information.

    Returns
    -------
    Atom
    """
    dictionary = copy.deepcopy(dictionary)
    atom_dict = {"atype": dictionary.pop("element")}
    atom_dict.update(dictionary)
    return Atom(**atom_dict)


def file_writer(filename, crystal_map, **kwargs):
    """Write a :class:`~orix.crystal_map.crystal_map.CrystalMap` object to
    an HDF5 file.

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

    file_dict = {
        "manufacturer": "orix",
        "version": __version__,
        "crystal_map": crystalmap2dict(crystal_map),
    }
    dict2hdf5group(file_dict, f["/"], **kwargs)

    f.close()


def crystalmap2dict(crystal_map, dictionary=None):
    """Get a dictionary from a
    :class:`~orix.crystal_map.crystal_map.CrystalMap` object with `data`
    and `header` keys with values.

    Parameters
    ----------
    crystal_map : CrystalMap
        Crystal map.
    dictionary : dict, optional
        Dictionary to update with crystal map information. If None
        (default), a new dictionary is created.

    Returns
    -------
    dictionary : dict
        Dictionary with crystal map information.
    """
    if dictionary is None:
        dictionary = {}

    # Get data cube coordinates in step size
    z, y, x = [
        0 if i is None else i for i in [crystal_map._z, crystal_map._y, crystal_map._x]
    ]
    # Get euler angles phi1, Phi, phi2
    eulers = crystal_map._rotations.to_euler()
    dictionary.update(
        {
            "data": {
                "z": z,
                "y": y,
                "x": x,
                "phi1": eulers[..., 0],
                "Phi": eulers[..., 1],
                "phi2": eulers[..., 2],
                "phase_id": crystal_map._phase_id,
                "id": crystal_map._id,
                "is_in_data": crystal_map.is_in_data,
            },
            "header": {
                "grid_type": "square",
                "nz": z.size if isinstance(z, np.ndarray) else 1,
                "ny": y.size if isinstance(y, np.ndarray) else 1,
                "nx": x.size if isinstance(x, np.ndarray) else 1,
                "z_step": crystal_map.dz,
                "y_step": crystal_map.dy,
                "x_step": crystal_map.dx,
                "rotations_per_point": crystal_map.rotations_per_point,
                "scan_unit": crystal_map.scan_unit,
            },
        }
    )
    dictionary["data"].update(crystal_map.prop)
    dictionary["header"].update({"phases": phaselist2dict(crystal_map.phases)})

    return dictionary


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
                    "The orix HDF5 writer could not write the following information to "
                    f"the file '{key} : {val}'."
                )
                break
        group.create_dataset(key, shape=dshape, dtype=ddtype, **kwargs)
        group[key][()] = val


def phaselist2dict(phases, dictionary=None):
    """Get a dictionary of phases.

    Parameters
    ----------
    phases : PhaseList
        Phases to write to file.
    dictionary : dict, optional
        Dictionary to update with information from multiple phases. If
        None (default), a new dictionary is created.

    Returns
    -------
    dictionary : dict
        Dictionary with information from multiple phases.
    """
    if dictionary is None:
        dictionary = {}
    dictionary.update({str(i): phase2dict(p) for i, p in phases})
    return dictionary


def phase2dict(phase, dictionary=None):
    """Get a dictionary of a phase.

    Parameters
    __________
    phase : Phase
        Phase to write to file.
    dictionary : dict, optional
        Dictionary to update with information from a single phase. If None
        (default), a new dictionary is created.

    Returns
    -------
    dictionary : dict
        Dictionary with information from a single phase.
    """
    if dictionary is None:
        dictionary = {}

    dictionary["name"] = phase.name
    if hasattr(phase.space_group, "number"):
        space_group = phase.space_group.number
    else:
        space_group = "None"
    if hasattr(phase.point_group, "name"):
        point_group = phase.point_group.name
    else:
        point_group = "None"
    dictionary["space_group"] = space_group
    dictionary["point_group"] = point_group
    dictionary["color"] = phase.color
    dictionary["structure"] = structure2dict(phase.structure)

    return dictionary


def structure2dict(structure, dictionary=None):
    """Get a dictionary of a phase's
    :class:`diffpy.structure.Structure` content.

    Only values necessary to initialize a structure object are returned.

    Parameters
    ----------
    structure : diffpy.structure.Structure
        Phase structure with a lattice and atoms.
    dictionary : dict, optional
        Dictionary to update with structure information. If None
        (default), a new dictionary is created.

    Returns
    -------
    dictionary : dict
        Dictionary with structure information.
    """
    if dictionary is None:
        dictionary = {}
    dictionary["lattice"] = lattice2dict(structure.lattice)
    atoms = structure.tolist()
    dictionary["atoms"] = {str(i): atom2dict(atom) for i, atom in enumerate(atoms)}
    return dictionary


def lattice2dict(lattice, dictionary=None):
    """Get a dictionary of a structure's
    :class:`diffpy.structure.Structure.lattice` content.

    Only values necessary to initialize a lattice object are returned.

    Parameters
    ----------
    lattice : diffpy.structure.Structure.lattice
        Structure lattice.
    dictionary : dict, optional
        Dictionary to update with structure lattice information. If None
        (default), a new dictionary is created.

    Returns
    -------
    dictionary : dict
        Dictionary with structure lattice information.
    """
    if dictionary is None:
        dictionary = {}
    dictionary["abcABG"] = np.array(lattice.abcABG())
    dictionary["baserot"] = lattice.baserot
    return dictionary


def atom2dict(atom, dictionary=None):
    """Get a dictionary of one of a structure's
    :class:`diffpy.structure.Structure.atoms` content.

    Only values necessary to initialize an atom object are returned.

    Parameters
    ----------
    atom : diffpy.structure.Structure.atom
        Atom in a structure.
    dictionary : dict, optional
        Dictionary to update with structure atom information. If None
        (default), a new dictionary is created.

    Returns
    -------
    dictionary : dict
        Dictionary with structure atoms information.
    """
    if dictionary is None:
        dictionary = {}
    dictionary.update(
        {
            attribute: atom.__getattribute__(attribute)
            for attribute in ["element", "label", "occupancy", "xyz", "U"]
        }
    )
    return dictionary
