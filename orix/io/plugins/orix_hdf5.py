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

"""Reader and writer of a crystal map to and from orix's own HDF5 file
format.
"""

import copy
from typing import Optional
from warnings import warn

from diffpy.structure import Atom, Lattice, Structure
from h5py import File, Group
import numpy as np

from orix.crystal_map import CrystalMap, Phase, PhaseList
from orix.io.plugins._h5ebsd import hdf5group2dict
from orix.quaternion import Rotation

__all__ = ["file_reader", "file_writer"]

# Plugin description
format_name = "orix_hdf5"
manufacturer = "orix"
file_extensions = ["h5", "hdf5"]
writes = True
writes_this = CrystalMap
# TODO: Extend reader/writer to Phase and PhaseList objects


def file_reader(filename: str, **kwargs) -> CrystalMap:
    """Return a crystal map from a file in orix's HDF5 file format.

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
    mode = kwargs.pop("mode", "r")
    with File(filename, mode=mode, **kwargs) as f:
        file_dict = hdf5group2dict(f["/"], recursive=True)
    return dict2crystalmap(file_dict["crystal_map"])


def dict2crystalmap(dictionary: dict) -> CrystalMap:
    """Get a crystal map from necessary items in a dictionary.

    Parameters
    ----------
    dictionary
        Dictionary with crystal map information.

    Returns
    -------
    xmap
        Crystal map.
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
    for direction in ["y", "x"]:
        this_direction = data.pop(direction)
        if hasattr(this_direction, "__iter__") is False:  # pragma: no cover
            this_direction = None
        crystal_map_dict[direction] = this_direction
    _ = [data.pop(i) for i in ["id"]]
    # What's left should be properties like quality metrics etc.
    crystal_map_dict.update({"prop": data})

    return CrystalMap(**crystal_map_dict)


def dict2phaselist(dictionary: dict) -> PhaseList:
    """Get a :class:`~orix.crystal_map.PhaseList` object from a
    dictionary.

    Parameters
    ----------
    dictionary
        Dictionary with phase list information.

    Returns
    -------
    phase_list
    """
    dictionary = copy.deepcopy(dictionary)
    return PhaseList(phases={int(k): dict2phase(v) for k, v in dictionary.items()})


def dict2phase(dictionary: dict) -> Phase:
    """Get a :class:`~orix.crystal_map.Phase` object from a dictionary.

    Parameters
    ----------
    dictionary
        Dictionary with phase information.

    Returns
    -------
    phase
    """
    dictionary = copy.deepcopy(dictionary)
    structure = dict2structure(dictionary["structure"])
    structure.title = dictionary["name"]
    space_group = dictionary["space_group"]  # Either "None" or int
    if space_group == "None":
        space_group = None
    else:
        space_group = int(space_group)
    point_group = dictionary["point_group"]
    if point_group == "None":
        point_group = None
    return Phase(
        name=dictionary["name"],
        space_group=space_group,
        point_group=point_group,
        structure=structure,
        color=dictionary["color"],
    )


def dict2structure(dictionary: dict) -> Structure:
    """Get a :class:`~diffpy.structure.Structure` object from a
    dictionary.

    Parameters
    ----------
    dictionary
        Dictionary with structure information.

    Returns
    -------
    structure
    """
    dictionary = copy.deepcopy(dictionary)
    return Structure(
        lattice=dict2lattice(dictionary["lattice"]),
        atoms=[dict2atom(atom) for atom in dictionary["atoms"].values()],
    )


def dict2lattice(dictionary: dict) -> Lattice:
    """Get a :class:`~diffpy.structure.Lattice` object from a
    dictionary.

    Parameters
    ----------
    dictionary
        Dictionary with lattice information.

    Returns
    -------
    lattice
    """
    dictionary = copy.deepcopy(dictionary)
    lattice_dict = {
        k: v
        for k, v in zip(["a", "b", "c", "alpha", "beta", "gamma"], dictionary["abcABG"])
    }
    lattice_dict["baserot"] = dictionary["baserot"]
    return Lattice(**lattice_dict)


def dict2atom(dictionary: dict) -> Atom:
    """Get a :class:`~diffpy.structure.Atom` object from a dictionary.

    Parameters
    ----------
    dictionary
        Dictionary with atom information.

    Returns
    -------
    atom
    """
    dictionary = copy.deepcopy(dictionary)
    atom_dict = {"atype": dictionary.pop("element")}
    atom_dict.update(dictionary)
    return Atom(**atom_dict)


def file_writer(filename: str, crystal_map: CrystalMap, **kwargs):
    """Write a crystal map to an HDF5 file.

    Parameters
    ----------
    filename
        Name of file to write to.
    crystal_map
        Object to write to file.
    **kwargs
        Keyword arguments passed to :meth:`h5py.Group.require_dataset`.
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


def crystalmap2dict(crystal_map: CrystalMap, dictionary: Optional[dict] = None) -> dict:
    """Get a dictionary from a :class:`~orix.crystal_map.CrystalMap`
    object with ``"data"`` and ``"header"`` keys with values.

    Parameters
    ----------
    crystal_map
        Crystal map.
    dictionary
        Dictionary to update with crystal map information. If not given
        (default), a new dictionary is created.

    Returns
    -------
    dictionary
        Dictionary with crystal map information.
    """
    if dictionary is None:
        dictionary = {}

    # Get data cube coordinates in step size
    y, x = [0 if i is None else i for i in [crystal_map._y, crystal_map._x]]
    # Get euler angles phi1, Phi, phi2
    eulers = crystal_map._rotations.to_euler()
    dictionary.update(
        {
            "data": {
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
                "ny": y.size if isinstance(y, np.ndarray) else 1,
                "nx": x.size if isinstance(x, np.ndarray) else 1,
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


def dict2hdf5group(dictionary: dict, group: Group, **kwargs):
    """Write a dictionary to datasets in a new group in an opened HDF5
    file.

    Parameters
    ----------
    dictionary
        Dataset names as keys with datasets as values.
    group
        HDF5 group to write dictionary to.
    **kwargs
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


def phaselist2dict(phases: PhaseList, dictionary: Optional[dict] = None) -> dict:
    """Get a dictionary of phases.

    Parameters
    ----------
    phases
        Phases to write to file.
    dictionary
        Dictionary to update with information from multiple phases. If
        not given (default), a new dictionary is created.

    Returns
    -------
    dictionary
        Dictionary with information from multiple phases.
    """
    if dictionary is None:
        dictionary = {}
    dictionary.update({str(i): phase2dict(p) for i, p in phases})
    return dictionary


def phase2dict(phase: Phase, dictionary: Optional[dict] = None) -> dict:
    """Get a dictionary of a phase.

    Parameters
    __________
    phase
        Phase to write to file.
    dictionary
        Dictionary to update with information from a single phase. If
        not given(default), a new dictionary is created.

    Returns
    -------
    dictionary
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


def structure2dict(structure: Structure, dictionary: Optional[dict] = None) -> dict:
    """Get a dictionary of a phase's
    :class:`~diffpy.structure.Structure` content.

    Only values necessary to initialize a structure object are returned.

    Parameters
    ----------
    structure
        Phase structure with a lattice and atoms.
    dictionary
        Dictionary to update with structure information. If not given
        (default), a new dictionary is created.

    Returns
    -------
    dictionary
        Dictionary with structure information.
    """
    if dictionary is None:
        dictionary = {}
    dictionary["lattice"] = lattice2dict(structure.lattice)
    atoms = structure.tolist()
    dictionary["atoms"] = {str(i): atom2dict(atom) for i, atom in enumerate(atoms)}
    return dictionary


def lattice2dict(lattice: Lattice, dictionary: Optional[dict] = None) -> dict:
    """Get a dictionary of a structure's
    :class:`~diffpy.structure.Structure.lattice` content.

    Only values necessary to initialize a lattice object are returned.

    Parameters
    ----------
    lattice
        Structure lattice.
    dictionary
        Dictionary to update with structure lattice information. If not
        given (default), a new dictionary is created.

    Returns
    -------
    dictionary
        Dictionary with structure lattice information.
    """
    if dictionary is None:
        dictionary = {}
    dictionary["abcABG"] = np.array(lattice.abcABG())
    dictionary["baserot"] = lattice.baserot
    return dictionary


def atom2dict(atom: Atom, dictionary: Optional[dict] = None) -> dict:
    """Get a dictionary of one of a structure's
    :class:`~diffpy.structure.Structure.atoms` content.

    Only values necessary to initialize an atom object are returned.

    Parameters
    ----------
    atom
        Atom in a structure.
    dictionary
        Dictionary to update with structure atom information. If not
        given (default), a new dictionary is created.

    Returns
    -------
    dictionary
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
