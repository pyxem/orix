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

""" This file contains functions (broadly internal ones) that support
the grid generation within rotation space """

import numpy as np

from orix.quaternion.rotation import Rotation


def uniform_SO3_sample(resolution,max_angle=None,old_method=False):
    """
    Returns rotations that are evenly spaced according to the Haar measure on
    SO3

    Parameters
    ----------
    resolution : float
        The characteristic distance between a rotation and its neighbour (degrees)
    max_angle : float
        The max angle (ie. distance from the origin) required from the gridding, (degrees)
    old_method : False
        Use the implementation adopted prior to version 0.6, offered for compatibility
    Returns
    -------
    q : orix.quaternion.rotation.Rotation
        grid containing appropriate rotations

    See Also
    --------
    orix.sample_generators.get_local_grid
    """
    if max_angle is not None and old_method:
        raise ValueError("old_method=True does not support using the max_angle keyword")
    #TODO: think more carefully about what resolution should mean

    num_steps = int(np.ceil(360 / resolution))
    if num_steps % 2 == 1:
        num_steps = int(num_steps + 1)
    half_steps = int(num_steps / 2)

    if not old_method:
        # sources can be found in the discussion of issue #175
        u_3 = np.linspace(0,1,num=num_steps,endpoint=True)

        if max_angle is None:
            u_1 = np.linspace(0,1,num=num_steps,endpoint=True)
            u_2 = np.linspace(0,1,num=num_steps,endpoint=True)
        else:
            # e_1 = cos(omega/2) = np.sqrt(1-u_1) * np.sin(2*np.pi*u2)
            e_1_max = np.cos(np.deg2rad(max_angle/2))
            u_1_max = 1 - np.square(e_1_max)
            u_2_max = np.arcsin(e_1_max) / 2 / np.pi
            u_1 = np.linspace(0,u_1_max,num=int(num_steps*u_1_max),endpoint=True)
            u_2 = np.linspace(0,u_2_max,num=int(num_steps*u_2_max),endpoint=True)

        # eyeballed this, will need checking
        inputs = np.meshgrid(u_1,u_2,u_3)

        # Convert u_1 etc. into the final form used
        a = np.sqrt(1-inputs[:,0])
        b = np.sqrt(inputs[:,0])
        s_2,c_2 = np.sin(2*np.pi*inputs[:,1]),np.cos(2*np.pi*inputs[:,1])
        s_3,c_3 = np.sin(2*np.pi*inputs[:,2]),np.cos(2*np.pi*inputs[:,2])

        q = np.asarray([a*s_2,a*c_2,b*s_3,b*c_3])

        # convert to quaternion object
        # remove duplicates

    if old_method:
        alpha = np.linspace(0, 2 * np.pi, num=num_steps, endpoint=False)
        beta = np.arccos(np.linspace(1, -1, num=half_steps, endpoint=False))
        gamma = np.linspace(0, 2 * np.pi, num=num_steps, endpoint=False)
        q = np.array(np.meshgrid(alpha, beta, gamma)).T.reshape((-1, 3))

        # convert to quaternions
        q = Rotation.from_euler(q, convention="bunge", direction="crystal2lab")

    # remove duplicates
    q = q.unique()

    return q
