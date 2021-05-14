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


def uniform_SO3_sample(resolution, max_angle=None, old_method=False):
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

    if old_method:
        return _euler_angles_harr_measure(resolution)
    else:
        return _three_uniform_samples_method(resolution, max_angle)

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

def _three_uniform_samples_method(resolution, max_angle):
    """
    Returns rotations that are evenly spaced according to the Haar measure on
    SO3, the advantage of this method is that it selects values from uniform distributions
    so we can more easily restrict to a subregion of SO3

    Parameters
    ----------
    resolution : float
        The characteristic distance between a rotation and its neighbour (degrees)
    max_angle : float
        The max angle (ie. distance from the origin) required from the gridding, (degrees)

    Returns
    -------
    q : orix.quaternion.rotation.Rotation
        grid containing appropriate rotations

    Notes
    -----
    This algorithm has a fairly light-footprint on the internet, it's implemented as
    described in [1], the 'gem' on which it is based can be found at [2] and
    has a reference [3]:

    [1] - http://planning.cs.uiuc.edu/node198.html
    [2] - http://inis.jinr.ru/sl/vol1/CMC/Graphics_Gems_3,ed_D.Kirk.pdf
    [3] - K. Shoemake. Uniform random rotations. Graphics Gems III, pages 124-132. Academic, New York, 1992.
    """
    # odd numbers of steps will not return 0.5 in the u_2 position
    num_steps = _resolution_to_num_steps(resolution,odd_only=True)

    if max_angle is None:
        u_1 = np.linspace(0, 1, num=num_steps, endpoint=True)
        u_2 = np.linspace(0, 1, num=num_steps, endpoint=False)
    else:
        raise ValueError("This method will not work")

    u_3 = np.linspace(0, 1, num=num_steps, endpoint=False)

    inputs = np.meshgrid(u_1, u_2, u_3)
    mesh1 = inputs[0].flatten()
    mesh2 = inputs[1].flatten()
    mesh3 = inputs[2].flatten()

    # Convert u_1 etc. into the final form used
    a = np.sqrt(1 - mesh1)
    b = np.sqrt(mesh1)
    s_2, c_2 = np.sin(2 * np.pi * mesh2), np.cos(2 * np.pi * mesh2)
    s_3, c_3 = np.sin(2 * np.pi * mesh3), np.cos(2 * np.pi * mesh3)

    q = np.asarray([a * s_2, a * c_2, b * s_3, b * c_3])

    # remove duplicates manually as this is memory efficient compared to Rotation.unique()

    # removes the double covering of quaternions
    q = np.where(q[0] > 0,+q,-q)
    # now rotation the same iff all elements are the same
    q = np.unique(q,axis=0)

    # convert to Rotation object
    q = Rotation(q.T)

    return q


def _euler_angles_harr_measure(resolution):
    """
    Returns rotations that are evenly spaced according to the Haar measure on
    SO3 using the euler angle parameterization

    Parameters
    ----------
    resolution : float
        The characteristic distance between a rotation and its neighbour (degrees)

    Returns
    -------
    q : orix.quaternion.rotation.Rotation
        grid containing appropriate rotations

    Notes
    -----
    The measures is proportional to cos(beta) dalpha dbeta dgamma ~
    see for example: https://math.stackexchange.com/questions/3316481/
    """

    num_steps = _resolution_to_num_steps(resolution,even_only=True)
    half_steps = int(num_steps / 2)

    alpha = np.linspace(0, 2 * np.pi, num=num_steps, endpoint=False)
    beta = np.arccos(np.linspace(1, -1, num=half_steps, endpoint=False))
    gamma = np.linspace(0, 2 * np.pi, num=num_steps, endpoint=False)
    q = np.array(np.meshgrid(alpha, beta, gamma)).T.reshape((-1, 3))

    # convert to quaternions
    q = Rotation.from_euler(q, convention="bunge", direction="crystal2lab")

    # remove duplicates
    q = q.unique()

    return q
