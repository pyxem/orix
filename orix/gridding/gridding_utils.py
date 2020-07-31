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

""" This file contains functions (broadly internal ones) that support
the grid generation within rotation space """

import numpy as np
from itertools import product

from diffpy.structure.spacegroups import GetSpaceGroup

from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C1, C2, C3, C4, C6, D2, D3, D4, D6, O, T

conversion_dict = {
    "PG1": C1,
    "PG1bar": C1,
    "PG2": C2,
    "PGm": C2,
    "PG2/m": C2,
    "PG222": D2,
    "PGmm2": C2,
    "PGmmm": D2,
    "PG4": C4,
    "PG4bar": C4,
    "PG4/m": C4,
    "PG422": D4,
    "PG4mm": C4,
    "PG4bar2m": D4,
    "PG4barm2": D4,
    "PG4/mmm": D4,
    "PG3": C3,
    "PG3bar": C3,
    "PG312": D3,
    "PG321": D3,
    "PG3m1": C3,
    "PG31m": C3,
    "PG3m": C3,
    "PG3bar1m": D3,
    "PG3barm1": D3,
    "PG3barm": D3,
    "PG6": C6,
    "PG6bar": C6,
    "PG6/m": C6,
    "PG622": D6,
    "PG6mm": C6,
    "PG6barm2": D6,
    "PG6bar2m": D6,
    "PG6/mmm": D6,
    "PG23": T,
    "PGm3bar": T,
    "PG432": O,
    "PG4bar3m": T,
    "PGm3barm": O,
}


def create_equispaced_grid(resolution):
    """
    Returns rotations that are evenly spaced according to the Harr measure on
    SO3

    Parameters
    ----------
    resolution : float
        The smallest distance between a rotation and its neighbour (degrees)

    Returns
    -------
    q : orix.Rotation
        grid containing appropriate rotations
    """
    num_steps = int(np.ceil(360 / resolution))

    alpha = np.linspace(0, 2*np.pi, num=num_steps, endpoint=False)
    beta = np.arccos(np.linspace(1, -1, num=num_steps, endpoint=False))
    gamma = np.linspace(0, 2*np.pi, num=num_steps, endpoint=False)
    q = np.asarray(list(product(alpha, beta, gamma)))

    # convert to quaternions
    q = Rotation.from_euler(q, convention="bunge", direction="crystal2lab")
    # remove duplicates
    q = q.unique()
    return q


def _get_proper_point_group(space_group_number):
    """
    Maps a space group number to a point group

    Parameters
    ----------
    space_group_number : int
        Between 1 and 231
        
    Returns
    -------
    point_group : orix.symmetry
        One of the 11 proper point groups
    """
    spg = GetSpaceGroup(space_group_number)
    pgn = spg.point_group_name

    return conversion_dict[pgn]
