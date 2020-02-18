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

"""Load and save utilities.

.. warning::

   These functions are far from complete or universally useful. Use at your
   own risk.

"""

import numpy as np


def loadang(file_string: str):
    """Load ``.ang`` files.

    Parameters
    ----------
    file_string : str
        Path to the ``.ang`` file. This file is assumed to list the Euler
        angles in the Bunge convention in the first three columns.

    Returns
    -------
    Rotation

    """
    from orix.quaternion.rotation import Rotation

    data = np.loadtxt(file_string)
    euler = data[:, :3]
    rotation = Rotation.from_euler(euler)
    return rotation


def loadctf(file_string: str):
    """Load ``.ang`` files.

    Parameters
    ----------
    file_string : str
        Path to the ``.ctf`` file. This file is assumed to list the Euler
        angles in the Bunge convention in the columns 5, 6, and 7.

    Returns
    -------
    Rotation

    """

    from orix.quaternion.rotation import Rotation

    data = np.loadtxt(file_string, skiprows=17)[:, 5:8]
    euler = np.radians(data)
    rotation = Rotation.from_euler(euler)
    return rotation
