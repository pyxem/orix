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

import matplotlib.colors as mcolors
import numpy as np


class AxisColorKey:
    def __init__(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @staticmethod
    def orientation2color(orientation):
        """From the NIST OOF2 package.
        https://github.com/usnistgov/OOF2/blob/master/SRC/engine/angle2color.C
        """
        axis = orientation.axis
        hue = axis.azimuth.data / (2 * np.pi)
        saturation = orientation.angle.data / np.pi
        costheta = axis.z.data / np.sqrt(axis.radial.data)
        value = 0.5 * (costheta + 1)
        hsv = np.column_stack([hue, saturation, value])
        return mcolors.hsv_to_rgb(hsv)
