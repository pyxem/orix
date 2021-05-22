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

"""Functions (broadly internal ones) supporting grid generation within
rotation space.
"""

import numpy as np

from orix.quaternion import Rotation


def uniform_SO3_sample(resolution, method="harr_euler", unique=True):
    r"""Returns rotations that are evenly spaced according to the Haar
    measure on *SO(3)*.

    Parameters
    ----------
    resolution : float
        The characteristic distance between a rotation and its neighbour
        in degrees.
    method : str
        The sampling method adopted, either "harr_euler" (default) or
        "quaternion". See *Notes*.
    unique : bool
        Whether only unique rotations should be returned, default is
        True.

    Returns
    -------
    q : orix.quaternion.Rotation
        Grid containing appropriate rotations.

    See Also
    --------
    :func:`orix.sampling.get_sample_local`
    :func:`orix.sampling.get_sample_fundamental`

    Notes
    -----
    The "quaternion" algorithm has a fairly light-footprint on the
    internet, it's implemented as described in [LaValle2006]_, the
    'gem' on which it is based can be found at [Kirk1995]_ and has a
    reference [Shoemake1992]_.

    The sample from the "harr_euler" algorithm is proportional to
    :math:`\cos(\beta) d\alpha \: d\beta \: d\gamma`. See for example:
    https://math.stackexchange.com/questions/3316481/.

    References
    ----------
    .. [LaValle2006] LaValle, Steven M. "Generating a random element of
        *SO(3)*," *From Planning Algorithms*, 2006,
        http://planning.cs.uiuc.edu/node198.html.
    .. [Kirk1995] Kirk, David. "Graphics Gems III," 1995,
        http://inis.jinr.ru/sl/vol1/CMC/Graphics_Gems_3,ed_D.Kirk.pdf.
    .. [Shoemake1992] Shoemake, Ken. "Uniform random rotations,"
        *Graphics Gems III (IBM Version)*, pp. 124-132, 1992.
    """
    if method == "harr_euler":
        return _euler_angles_harr_measure(resolution, unique)
    elif method == "quaternion":
        return _three_uniform_samples_method(resolution, unique)


def _three_uniform_samples_method(resolution, unique, max_angle=None):
    """Returns rotations that are evenly spaced according to the Haar
    measure on *SO(3)*. The advantage of this method compared to
    :func:`_euler_angles_harr_measure` is that it selects values from
    uniform distributions so that we can more easily restrict to a
    subregion of *SO(3)*.

    Parameters
    ----------
    resolution : float
        The characteristic distance between a rotation and its neighbour
        in degrees.
    unique : bool
        Whether only unique rotations should be returned.
    max_angle : float
        Only rotations with angles smaller than max_angle are returned

    Returns
    -------
    q : orix.quaternion.Rotation
        Grid containing appropriate rotations.

    Notes
    -----
    See *Notes* in :func:`uniform_SO3_sample`.
    """
    num_steps = _resolution_to_num_steps(resolution)

    if max_angle is not None:
        # e_1 = cos(omega/2) = np.sqrt(1-u_1) * np.sin(2*np.pi*u2)
        e_1_min = np.cos(np.deg2rad(max_angle / 2))
        u_1_max = 1 - np.square(e_1_min)
        u_2_min = np.arcsin(e_1_min) / 2 / np.pi

        # round these up to avoid zero selection
        num_1 = int(num_steps * (u_1_max) + 0.5)
        num_2 = int(num_steps * (1 - u_2_min) + 0.5)
        u_1 = np.linspace(0, u_1_max, num=num_1, endpoint=True)
        u_2 = np.linspace(u_2_min, 1, num=num_2, endpoint=True)

    else:
        u_1 = np.linspace(0, 1, num=num_steps, endpoint=True)
        u_2 = np.linspace(0, 1, num=num_steps, endpoint=False)

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
    q = Rotation(q.T)

    if unique:
        q = q.unique()

    return q


def _euler_angles_harr_measure(resolution, unique):
    """Returns rotations that are evenly spaced according to the Haar
    measure on *SO(3)* using the Euler angle parameterization.

    Parameters
    ----------
    resolution : float
        The characteristic distance between a rotation and its neighbour
        in degrees.
    unique : bool
        Whether only unique rotations should be returned.

    Returns
    -------
    q : orix.quaternion.Rotation
        Grid containing appropriate rotations.

    Notes
    -----
    See *Notes* in :func:`uniform_SO3_sample`.
    """
    num_steps = _resolution_to_num_steps(resolution, even_only=True)
    half_steps = int(num_steps / 2)

    alpha = np.linspace(0, 2 * np.pi, num=num_steps, endpoint=False)
    beta = np.arccos(np.linspace(1, -1, num=half_steps, endpoint=False))
    gamma = np.linspace(0, 2 * np.pi, num=num_steps, endpoint=False)
    q = np.array(np.meshgrid(alpha, beta, gamma)).T.reshape((-1, 3))

    # Convert to quaternions
    q = Rotation.from_euler(q, convention="bunge", direction="crystal2lab")

    if unique:
        q = q.unique()

    return q


def _resolution_to_num_steps(resolution, even_only=False, odd_only=False):
    """Convert a 'resolution' to number of steps when sampling
    orientations on a linear axis.

    Parameters
    ----------
    resolution : float
        Characteristic distance between a rotation and its neighbour in
        degrees.
    even_only : bool, optional
        Force the returned `num_steps` to be even, default is False.
    odd_only : bool, optional
        Force the returned `num_steps` to be odd, defaults is False.

    Returns
    -------
    num_steps : int
        Number of steps to use when sampling a 'full' linear axes.
    """
    num_steps = int(np.ceil(360 / resolution))
    modulo2 = num_steps % 2
    if (even_only and modulo2 == 1) or (odd_only and modulo2 == 0):
        num_steps += 1
    return num_steps
