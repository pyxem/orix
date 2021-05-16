# -*- coding: utf-8 -*-
# Copyright 2018-2021 the orix developers
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

import numpy as np

def _resolution_to_num_steps(resolution,even_only=False,odd_only=False):
    """ Converts a user input resolution to a number off steps (ie. on a linear axis)

    Parameters
    ----------
    resolution : float
        The characteristic distance between a rotation and its neighbour (degrees)
    even_only : bool, optional
        Force the returned num_steps to be even, defaults False
    odd_only : bool, optional
        Force the returned num_steps to be odd, defaults False

    Returns
    -------
    num_steps : int
        The number of steps to use sampling a 'full' linear axes
    """
    num_steps = int(np.ceil(360 / resolution))

    if even_only:
        if num_steps % 2 == 1:
            num_steps = int(num_steps + 1)

    elif odd_only:
            if num_steps % 2 == 0:
                num_steps = int(num_steps + 1)

    return num_steps
