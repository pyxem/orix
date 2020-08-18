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

""" This file contains functions (broadly internal ones) that support
the grid generation within rotation space """

import numpy as np
from itertools import product

from orix.quaternion.rotation import Rotation


def uniform_SO3_sample(resolution):
    """
    Returns rotations that are evenly spaced according to the Haar measure on
    SO3

    Parameters
    ----------
    resolution : float
        The characteristic distance between a rotation and its neighbour (degrees)

    Returns
    -------
    q : orix.quaternion.rotation.Rotation
        grid containing appropriate rotations
    """
    num_steps = int(np.ceil(360 / resolution))
    if num_steps % 2 == 1:
        num_steps = int(num_steps + 1)
    half_steps = int(num_steps / 2)

    alpha = np.linspace(0, 2 * np.pi, num=num_steps, endpoint=False)
    beta = np.arccos(np.linspace(1, -1, num=half_steps, endpoint=False))
    gamma = np.linspace(0, 2 * np.pi, num=num_steps, endpoint=False)
    q = np.asarray(list(product(alpha, beta, gamma)))

    # convert to quaternions
    q = Rotation.from_euler(q, convention="bunge", direction="crystal2lab")
    # remove duplicates
    q = q.unique()
    return q
