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

from orix.quaternion.rotation import Rotation
from orix.quaternion.symmetry import C1,C2,C3,C4,C6,D2,D3,D4,D6,O,T


def create_equispaced_grid(resolution):
    """
    Returns rotations that are evenly spaced according to the Harr measure on
    SO3

    Parameters
    ----------

    Returns
    -------
    """
    num_steps = int(np.ceil(360 / resolution))

    alpha = np.linspace(0, np.pi, num=num_steps, endpoint=False)
    beta = np.arcos(np.linspace(1, -1, num=num_steps, endpoint=False))
    gamma = np.linspace(0, np.pi, num=num_steps, endpoint=False)
    q = np.asarray(list(product(alpha, beta, gamma)))

    # convert to quaternions
    q = Rotation.from_euler(q, convention="bunge", direction="crystal2lab")
    # remove duplicates
    q = q.unique()
    return q


def get_proper_point_group(space_group_number):
    """
    Maps a space-group-number to a point group

    Parameters
    ----------
    space_group_number : int

    Returns
    -------
    point_group :

    Notes
    -----
    This function enumerates the list on https://en.wikipedia.org/wiki/List_of_space_groups
    Point groups (32) are then converted to proper point groups (11) using the Schoenflies
    representations given in that table.
    """

    if space_group_number in [1, 2]:
        return C1  # triclinic
    if 2 < space_group_number < 16:
        return C2  # monoclinic
    if 15 < space_group_number < 75:
        return D2  # orthorhomic
    if 74 < space_group_number < 143:  # tetragonal
        if (74 < space_group_number < 89) or (99 < space_group_number < 110):
            return C4
        else:
            return D4
    if 142 < space_group_number < 168:  # trigonal
        if 142 < space_group_number < 148 or 156 < space_group_number < 161:
            return C3
        else:
            return D3
    if 167 < space_group_number < 194:  # hexagonal
        if 167 < space_group_number < 176 or space_group_number in [183, 184, 185, 186]:
            return C6
        else:
            return D6
    if 193 < space_group_number < 231:  # cubic
        if 193 < space_group_number < 207 or space_group_number in [
            215,
            216,
            217,
            218,
            219,
            220,
        ]:
            return O  # oct
        else:
            return T  # tet
