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

from matplotlib.axes import Axes
from matplotlib import axis
from matplotlib.path import Path
from matplotlib.ticker import NullLocator
from matplotlib.patches import Circle
from matplotlib.transforms import Affine2DBase, Affine2D, BboxTransformTo, \
    Transform
from matplotlib.spines import Spine

from matplotlib.projections import register_projection

import numpy as np


class StereographicTransform(Transform):
    """
    Basic stereographic transformation.
    1. `phi, theta` -> `R, theta` : spherical polar to plane polar coordinates
    2. `R, theta` -> `x, y` : plane polar to plane cartesian coordinates
    """
    input_dims = 2
    output_dims = 2
    is_separable = False

    def __init__(self, axis=None):
        Transform.__init__(self)
        self._axis = axis

    def transform_non_affine(self, values):
        xy = np.empty(values.shape, float)
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        phi = np.pi - values[:, 0:1]
        theta = values[:, 1:2]
        r = np.sin(phi) / (1 - np.cos(phi))
        x[:] = r * np.cos(theta)
        y[:] = r * np.sin(theta)
        return xy

    transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

    def transform_path_non_affine(self, path):
        ipath = path.interpolated(path._interpolation_steps)
        return Path(self.transform(ipath.vertices), ipath.codes)

    transform_path_non_affine.__doc__ = Transform.transform_path_non_affine.__doc__

    def inverted(self):
        return InvertedStereographicTransform(self._axis)

    inverted.__doc__ = Transform.inverted.__doc__


class InvertedStereographicTransform(Transform):
    """
    The stereographic transformation, inverted.
    1. `x, y` -> `R, theta` : plane Cartesian to plane polar coordinates
    2. `R, theta` -> `phi, theta` : plane polar to spherical polar coordinates
    Not to be confused with the inverse stereographic transformation.
    """
    input_dims = 2
    output_dims = 2
    is_separable = False

    def __init__(self, axis=None):
        Transform.__init__(self)
        self._axis = axis

    def transform_non_affine(self, values):
        phitheta = np.empty(values.shape, float)
        phi = phitheta[:, 0:1]
        theta = phitheta[:, 1:2]
        x = values[:, 0:1]
        y = values[:, 1:2]
        c = x + 1j * y
        r = np.absolute(c)
        theta[:] = np.angle(c)
        phi[:] = np.pi - 2 * np.arctan(1 / r)
        return phitheta


class StereographicAffine(Affine2DBase):

    def get_matrix(self):
        if self._invalid:
            transform = StereographicTransform()
            xscale, _ = transform.transform_point((np.pi / 2, 0))
            _, yscale = transform.transform_point((np.pi / 2, np.pi / 2))
            affine = Affine2D().scale(0.5 / xscale, 0.5 / yscale).translate(0.5,
                                                                            0.5)
            self._mtx = affine.get_matrix()
            self._inverted = None
            self._invalid = 0
        return self._mtx

    get_matrix.__doc__ = Affine2DBase.get_matrix.__doc__


class StereographicAxes(Axes):

    name = 'stereographic'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_aspect('equal')
        self.cla()

    def _init_axis(self):
        # Need to override these to get rid of spines.
        self.xaxis = axis.XAxis(self)
        self.yaxis = axis.YAxis(self)

    def cla(self):
        # Default values for phi and theta "ticks" (for the grid)
        # Turn off actual axis ticks (and labels)
        # Set the axis limits
        Axes.cla(self)
        self.set_xticks([0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])
        self.set_yticks(np.linspace(0, 2 * np.pi, 13))
        self.xaxis.set_minor_locator(NullLocator())
        self.yaxis.set_minor_locator(NullLocator())
        self.xaxis.set_ticks_position('none')
        self.xaxis.set_ticks_position('none')
        self.xaxis.set_tick_params(tick1On=False, label1On=False)
        self.yaxis.set_tick_params(tick1On=False, label1On=False)
        self.set_xlim(0, np.pi / 2)
        self.set_ylim(0, 2 * np.pi)

    def _set_lim_and_transforms(self):
        self.transProjection = StereographicTransform()
        self.transAffine = StereographicAffine()
        self.transAxes = BboxTransformTo(self.bbox)

        self.transData = self.transProjection + self.transAffine + self.transAxes
        self._xaxis_pretransform = Affine2D().scale(1., 2 * np.pi)
        self._xaxis_transform = self._xaxis_pretransform + self.transData
        self._yaxis_pretransform = Affine2D().scale(np.pi / 2, 1.)
        self._yaxis_transform = self._yaxis_pretransform + self.transData

    def get_xaxis_transform(self, which='grid'):
        # Need to override this to get rid of spines.
        return self._xaxis_transform

    def get_yaxis_transform(self, which='grid'):
        # Need to override this to get rid of spines.
        return self._yaxis_transform

    def format_coord(self, phi, theta):
        # Spherical polar coordinates in degrees.
        phi = np.rad2deg(phi)
        theta = np.rad2deg(theta)
        return '{:.2f}\u00b0, {:.2f}\u00b0'.format(phi, theta)

    def _gen_axes_spines(self):
        """
        Bordering spine.
        """
        return {'polar': Spine.circular_spine(self, (0.5, 0.5), 0.5)}

    def _gen_axes_patch(self):
        return Circle((0.5, 0.5), 0.5)

    def get_data_ratio(self):
        # Enforces the patch to remain circular.
        return 1.

    def can_zoom(self):
        return False

    def can_pan(self):
        return False

    def transform(self, xs):
        from texpy.quaternion.rotation import Rotation
        from texpy.vector import Vector3d
        if isinstance(xs, Rotation):
            x, y, z = (xs * Vector3d.zvector()).xyz
        else:
            x, y, z = Vector3d(xs).unit.xyz
        phi = np.arcsin(np.sqrt(x ** 2 + y ** 2))
        theta = np.angle(x + 1j * y)
        return phi, theta

    def scatter(self, xs, **kwargs):
        phi, theta = self.transform(xs)
        super().scatter(phi, theta, **kwargs)

register_projection(StereographicAxes)
