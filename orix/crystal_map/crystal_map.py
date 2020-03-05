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

import numpy as np

from orix.quaternion.rotation import Rotation
from orix.quaternion.orientation import Orientation
from orix.quaternion.symmetry import _groups


class CrystalMap:
    """A crystallographic map storing measurement pixels as a 2D array with
    rows containing the orientation, the spatial coordinates, the phase and
    potential properties of each individual pixel.

    Attributes
    ----------
    id : numpy.ndarray
        Unique ID of each pixel.
    rotations : orix.quaternion.rotation.Rotation
        Rotation of each pixel.
    orientations : orix.quaternion.orientation.Orientation
        Orientation of each pixel.
    mineral : list of str
        Name of phases.
    phase_id : numpy.ndarray
        Phase ID of each pixel.
    phase : numpy.ndarray
        Phase ID of each pixel as imported.
    props : dict
        Dictionary of numpy arrays of quality metrics or other auxiliary
        properties of each pixel.
    x, y, z : numpy.ndarray
        Coordinates of the centre of each pixel.
    crystal_symmetries : list of orix.symmetry.Symmetry
        Crystal symmetries of phases in the map. Symmetries in this list
        map directly to phases in the phase_id list.
    """
    def __init__(
            self,
            rotations,
            phase_id=None,
            phase=None,
            x=None,
            y=None,
            z=None,
            properties=None,
            crystal_symmetries=None,
    ):
        self.id = np.arange(len(rotations))
        self.rotations = Rotation.from_euler(rotations)

        # Set auxiliary properties
        self.props = properties

        # Set pixel coordinates
        if x is None:
            x = np.arange(len(rotations))
        if y is None:
            y = np.zeros_like(x)
        if z is None:
            z = np.zeros_like(x)
        self.x = x
        self.y = y
        self.z = z

        # Set phases as imported
        if phase is None:
            phase = np.zeros(len(rotations))
        self.phase = phase

        # Set phase ID, which maps to entries in crystal_symmetries
        if phase_id is None:
            phase_id = np.zeros(len(rotations))
        self.phase_id = phase_id

        # Set crystal symmetry
        self.crystal_symmetries = []
        if crystal_symmetries is not None:
            for cs_in in crystal_symmetries:
                if cs_in == '43':
                    cs_in = '432'
                for cs in _groups:
                    if cs_in == cs.name.replace('-', ''):
                        break
                    else:
                        cs = _groups[0]
                self.crystal_symmetries.append(cs)

        # Set orientations if crystal symmetries were passed
#        self.orientations =
#            self.orientations = Orientation.from_euler(rotations).set_symmetry(
#                crystal_symmetry
#            )
#        else:
#            self.orientations = None

        # Set properties
        self.prop = properties

    def __repr__(self):
        header = "Phase\tOrientations\tMineral\tSymmetry"
        return header
