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

from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.axis import XAxis, YAxis
from matplotlib.patches import Circle
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.ticker import NullLocator, FixedLocator
from matplotlib.transforms import Affine2D, Affine2DBase, BboxTransformTo, Transform
import numpy as np

from orix.projections import InverseStereographicProjection, StereographicProjection
from orix.quaternion import Rotation
from orix.vector import Vector3d


class StereographicTransform(Transform):
    input_dims = output_dims = 2

    def transform_non_affine(self, values):
        """(azimuthal, polar) to (X, Y)."""
        azimuthal, polar = values.T
        x, y = StereographicProjection().spherical2xy(polar=polar, azimuthal=azimuthal)
        return np.column_stack([x, y])

    def transform_path_non_affine(self, path):
        ipath = path.interpolated(path._interpolation_steps)
        return Path(self.transform(ipath.vertices), ipath.codes)

    def inverted(self):
        return InverseStereographicTransform()


class InverseStereographicTransform(Transform):
    input_dims = output_dims = 2

    def transform_non_affine(self, values):
        """(X, Y) to (azimuthal, polar)."""
        x, y = values.T
        azimuthal, polar = InverseStereographicProjection().xy2spherical(x=x, y=y)
        return np.column_stack([azimuthal, polar])

    def inverted(self):
        return StereographicTransform()


class StereographicAffine(Affine2DBase):
    def get_matrix(self):
        st = StereographicTransform()
        xscale, _ = st.transform((0, np.pi / 2))
        _, yscale = st.transform((np.pi / 2, np.pi / 2))
        scales = (0.5 / xscale, 0.5 / yscale)
        return Affine2D().scale(*scales).translate(0.5, 0.5)


class StereographicAxes(Axes):
    """Stereographic projection."""

    name = "stereographic"

    def __init__(self, *args, **kwargs):
        self._polar_cap = np.pi / 2
        self._azimuthal_cap = 2 * np.pi
        super().__init__(*args, **kwargs)
        # Set ratio of y-unit to x-unit by adjusting the physical
        # dimension of the Axes (box), and centering the anchor (C)
        self.set_aspect("equal", adjustable="box", anchor="C")
        self.cla()

    def _init_axis(self):
        # Need to override these to get rid of spines
        self.xaxis = XAxis(self)
        self.yaxis = YAxis(self)
        self.spines["stereographic"].register_axis(self.yaxis)
        self._update_transScale()

    def cla(self):
        """Resetting of the axes."""
        super().cla()

        self.xaxis.set_ticks_position("none")
        self.yaxis.set_ticks_position("none")
        self.xaxis.set_tick_params(label1On=False)
        self.yaxis.set_tick_params(label1On=False)

        resolution = 15
        self.set_polar_grid(resolution)
        self.set_azimuthal_grid(resolution)
        self.grid(rcParams["axes.grid"])  # Boolean

    def _set_lim_and_transforms(self):
        self.transProjection = StereographicTransform()
        self.transAffine = StereographicAffine()
        self.transAxes = BboxTransformTo(self.bbox)

        self.transData = self.transProjection + self.transAffine + self.transAxes

        self._xaxis_pretransform = Affine2D().scale(1, self._polar_cap)
        self._xaxis_transform = self._xaxis_pretransform + self.transData

        self._yaxis_pretransform = Affine2D().scale(self._azimuthal_cap, 1)
        self._yaxis_transform = self._yaxis_pretransform + self.transData

    def set_polar_grid(self, resolution):
        resolution = np.deg2rad(resolution)
        grid = np.arange(0, self._polar_cap, resolution)
        self.set_yticks(grid)

    def set_azimuthal_grid(self, resolution):
        resolution = np.deg2rad(resolution)
        grid = np.arange(0, self._azimuthal_cap, resolution)
        self.set_xticks(grid)

    @staticmethod
    def format_coord(azimuthal, polar):
        azimuthal_deg = np.rad2deg(azimuthal)
        polar_deg = np.rad2deg(polar)
        return (
            "\N{GREEK SMALL LETTER THETA}={:.2f}\N{GREEK SMALL LETTER PI} "
            "({:.2f}\N{DEGREE SIGN}), "
            "\N{GREEK SMALL LETTER PHI}={:.2f}\N{GREEK SMALL LETTER PI} "
            "({:.2f}\N{DEGREE SIGN})"
        ).format(azimuthal, azimuthal_deg, polar, polar_deg)

    def get_xaxis_transform(self, which="grid"):
        # Need to override this to get rid of spines.
        return self._xaxis_transform

    def get_yaxis_transform(self, which="grid"):
        # Need to override this to get rid of spines.
        return self._yaxis_transform

    def _gen_axes_spines(self):
        return {"stereographic": Spine.circular_spine(self, (0.5, 0.5), 0.5)}

    def _gen_axes_patch(self):
        return Circle((0.5, 0.5), 0.5)

    def get_data_ratio(self):
        return 1

    def can_pan(self):
        return False

    def can_zoom(self):
        return False

    @staticmethod
    def _pretransform_input(values):
        if len(values) == 2:
            azimuthal, polar = values
        else:
            value = values[0]
            if isinstance(value, Rotation):
                v = value * Vector3d.zvector()
                azimuthal = v.phi.data
                polar = v.theta.data
            elif isinstance(value, Vector3d):
                azimuthal = value.phi.data
                polar = value.theta.data
            else:
                raise ValueError(
                    "Accepts only one, Vector3d or Rotation, or two, (azimuthal, "
                    "polar), input arguments"
                )
        return azimuthal, polar

    def scatter(self, *args, **kwargs):
        azimuthal, polar = self._pretransform_input(args)
        super().scatter(azimuthal, polar, **kwargs)


register_projection(StereographicAxes)
