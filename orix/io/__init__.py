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
    from texpy.quaternion.rotation import Rotation
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

    from texpy.quaternion.rotation import Rotation
    data = np.loadtxt(file_string, skiprows=17)[:, 5:8]
    euler = np.radians(data)
    rotation = Rotation.from_euler(euler)
    return rotation

