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

from orix.plot.orientation_color_keys import OrientationColorKey


class RGBColorKey(OrientationColorKey):
    def orientation2color(self, orientations, scale=True):
        alpha, beta, gamma = orientations.to_euler_in_fundamental_region().T
        max_alpha, max_beta, max_gamma = self.symmetry.max_euler_angles
        r = alpha / max_alpha
        g = beta / max_beta
        b = gamma / max_gamma
        rgb = np.column_stack([r, g, b])
        if scale:
            rgb /= rgb.max(axis=1)[:, np.newaxis]
        return rgb
