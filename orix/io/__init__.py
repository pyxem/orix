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

"""Read and write crystal maps from and to file.

.. currentmodule:: orix.io

.. rubric:: Modules

.. autosummary::
    :toctree: ../generated/
    :template: custom-module-template.rst

    plugins
"""

import os
from pathlib import Path
from typing import Optional, Union
from warnings import warn

from h5py import File, is_hdf5
import numpy as np

from orix._util import deprecated
from orix.crystal_map import CrystalMap
from orix.io.plugins import plugin_list
from orix.io.plugins._h5ebsd import hdf5group2dict
from orix.quaternion import Rotation

extensions = [plugin.file_extensions for plugin in plugin_list if plugin.writes]


__all__ = [
    "loadang",
    "loadctf",
    "load",
    "save",
]


# TODO: Remove after 0.13.0
@deprecated(since="0.13", removal="0.14", alternative="io.load")
def loadang(file_string: str) -> Rotation:
    """Load ``.ang`` files.

    Parameters
    ----------
    file_string
        Path to the ``.ang`` file. This file is assumed to list the
        Euler angles in the Bunge convention in the first three columns.

    Returns
    -------
    rotation
        Rotations in the file.
    """
    data = np.loadtxt(file_string)
    euler = data[:, :3]
    return Rotation.from_euler(euler)


# TODO: Remove after 0.13.0
@deprecated(since="0.13", removal="0.14", alternative="io.load")
def loadctf(file_string: str) -> Rotation:
    """Load ``.ctf`` files.

    Parameters
    ----------
    file_string
        Path to the ``.ctf`` file. This file is assumed to list the
        Euler angles in the Bunge convention in the columns 5, 6, and 7.

    Returns
    -------
    rotation
        Rotations in the file.
    """
    data = np.loadtxt(file_string, skiprows=17)[:, 5:8]
    euler = np.radians(data)
    return Rotation.from_euler(euler)


def load(filename: Union[str, Path], **kwargs) -> CrystalMap:
    """Load data from a supported file format listed in
    :doc:`orix.io.plugins`.

    Parameters
    ----------
    filename
        Name of file to load.
    **kwargs
        Keyword arguments passed to the corresponding plugins'
        ``file_reader()``. See their individual docstrings for available
        arguments.

    Returns
    -------
    data
        Crystal map read from the file.
    """
    if not os.path.isfile(filename):
        raise IOError(f"No filename matches '{filename}'.")

    # Find matching reader for file extension
    extension = os.path.splitext(filename)[1][1:]
    readers = []
    for plugin in plugin_list:
        if extension.lower() in plugin.file_extensions:
            readers.append(plugin)

    n_matching_readers = len(readers)
    if n_matching_readers == 0:
        raise IOError(
            f"Could not read '{filename}'. If the file format is supported, please "
            "report this error."
        )
    elif n_matching_readers > 1 and is_hdf5(filename):
        reader = _plugin_from_manufacturer(filename, readers)
    else:
        reader = readers[0]

    return reader.file_reader(filename, **kwargs)


def _plugin_from_manufacturer(filename: str, plugins: list):
    """Return the correct plugin based on the manufacturer listed in a
    top group named 'Manufacturer' in an HDF5 file.

    Parameters
    ----------
    filename
        Name of the file to find the correct plugin for.
    plugins
        List of potential HDF5 plugins.

    Returns
    -------
    matching_plugin
        One of the potential plugins, or None if no matching plugin was
        found.
    """
    with File(filename) as f:
        d = hdf5group2dict(f["/"])
        manufacturer = None
        for key, value in d.items():
            if key.lower() == "manufacturer":
                manufacturer = value
        matching_plugin = None
        for p in plugins:
            if (
                hasattr(p, "manufacturer")
                and manufacturer is not None
                and p.manufacturer in manufacturer
            ):
                matching_plugin = p
    return matching_plugin


def save(
    filename: Union[str, Path],
    object2write: CrystalMap,
    overwrite: Optional[bool] = None,
    **kwargs,
):
    """Write data to a supported file format listed in
    :doc:`orix.io.plugins`.

    Parameters
    ----------
    filename
        Name of file to write to.
    object2write
        Object to write to file.
    overwrite
        If not given and the file exists, the user is queried. If
        ``True`` (``False``) the file is (not) overwritten if it exists.
    **kwargs
        Keyword arguments passed to the corresponding plugins'
        ``file_writer()``. See their individual docstrings for available
        arguments.
    """
    ext = os.path.splitext(filename)[1][1:]

    writer = None
    for p in plugin_list:
        if (
            ext.lower() in p.file_extensions
            and p.writes
            and isinstance(object2write, p.writes_this)
        ):
            writer = p
            break

    if writer is None:
        raise IOError(
            f"'{ext}' does not correspond to any supported format. Supported "
            f"file extensions are: '{extensions}'."
        )
    else:
        is_file = os.path.isfile(filename)
        if overwrite is None:
            write = _overwrite_or_not(filename)  # Ask what to do
        elif overwrite is True or (overwrite is False and not is_file):
            write = True
        elif overwrite is False and is_file:
            write = False
        else:
            raise ValueError("`overwrite` parameter can only be None, True or False.")

    if write:
        writer.file_writer(filename, object2write, **kwargs)


def _overwrite_or_not(filename: str) -> bool:
    """If the file exists, ask the user for overwriting and return
    ``True`` or ``False``, else return ``True``.

    This function is adapted from HyperSpy.

    Parameters
    ----------
    filename
        Name of file to write to.

    Returns
    -------
    overwrite
        Whether to overwrite the file.
    """
    overwrite = True
    if os.path.isfile(filename):
        message = "Overwrite '%s' (y/n)?\n" % filename
        try:
            answer = input(message).lower()
            while (answer != "y") and (answer != "n"):
                print("Please answer y or n.")
                answer = input(message).lower()
            if answer == "n":
                overwrite = False
        except OSError:
            warn(
                "Not overwriting, since your terminal does not support raw input. To "
                "overwrite the file, use `overwrite=True`."
            )
            overwrite = False
    return overwrite
