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
from matplotlib.transforms import Affine2D, Affine2DBase, BboxTransformTo, Transform
import numpy as np

from orix.projections import InverseStereographicProjection, StereographicProjection
from orix.quaternion import Rotation
from orix.vector import Vector3d


class StereographicTransform(Transform):
    input_dims = output_dims = 2

    def __init__(self, pole=-1):
        super().__init__()
        self.pole = pole

    def transform_non_affine(self, values):
        """(Azimuth, polar) to (X, Y)."""
        azimuth, polar = values.T
        x, y = StereographicProjection(pole=self.pole).spherical2xy(
            azimuth=azimuth, polar=polar
        )
        return np.column_stack([x, y])

    def transform_path_non_affine(self, path):
        ipath = path.interpolated(path._interpolation_steps)
        return Path(self.transform(ipath.vertices), ipath.codes)

    def inverted(self):
        return InverseStereographicTransform(pole=self.pole)


class InverseStereographicTransform(Transform):
    input_dims = output_dims = 2

    def __init__(self, pole=-1):
        super().__init__()
        self.pole = pole

    def transform_non_affine(self, values):
        """(X, Y) to (azimuth, polar)."""
        x, y = values.T
        azimuth, polar = InverseStereographicProjection(pole=self.pole).xy2spherical(
            x=x, y=y
        )
        return np.column_stack([azimuth, polar])

    @staticmethod
    def inverted():
        return StereographicTransform()


class StereographicAffine(Affine2DBase):
    def __init__(self, pole=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pole = pole

    def get_matrix(self):
        st = StereographicTransform(pole=self.pole)
        xscale, _ = st.transform((0, np.pi / 2))
        _, yscale = st.transform((np.pi / 2, np.pi / 2))
        scales = (0.5 / xscale, 0.5 / yscale)
        return Affine2D().scale(*scales).translate(0.5, 0.5)


class StereographicPlot(Axes):
    """Stereographic projection."""

    name = "stereographic"
    _pole = -1

    def __init__(self, *args, **kwargs):
        self._polar_cap = np.pi / 2
        self._polar_resolution = 30

        self._azimuth_cap = 2 * np.pi
        self._azimuth_resolution = 30

        super().__init__(*args, **kwargs)
        # Set ratio of y-unit to x-unit by adjusting the physical
        # dimension of the Axes (box), and centering the anchor (C)
        self.set_aspect("equal", adjustable="box", anchor="C")
        self.clear()

    def _init_axis(self):
        # Need to override these to get rid of spines
        self.xaxis = XAxis(self)
        self.yaxis = YAxis(self)
        self.spines["stereographic"].register_axis(self.yaxis)
        self._update_transScale()

    def clear(self):
        """Resetting of the axes."""
        super().clear()

        self.xaxis.set_ticks_position("none")
        self.yaxis.set_ticks_position("none")
        self.xaxis.set_tick_params(label1On=False)
        self.yaxis.set_tick_params(label1On=False)

        self.set_polar_grid()
        self.set_azimuth_grid()
        self.grid(rcParams["axes.grid"])

        self.set_xlim(0, self._azimuth_cap)
        self.set_ylim(0, self._polar_cap)

    def _set_lim_and_transforms(self):
        self.transShift = Affine2D().scale(-self._pole, 1)
        self.transProjection = StereographicTransform(pole=self.pole)
        self.transAffine = StereographicAffine(pole=self.pole)
        self.transAxes = BboxTransformTo(self.bbox)

        self.transData = (
            self.transShift + self.transProjection + self.transAffine + self.transAxes
        )

        self._xaxis_pretransform = Affine2D().scale(1, self._polar_cap)
        self._xaxis_transform = self._xaxis_pretransform + self.transData

        self._yaxis_pretransform = Affine2D().scale(self._azimuth_cap, 1)
        self._yaxis_transform = self._yaxis_pretransform + self.transData

    @staticmethod
    def format_coord(azimuth, polar):
        azimuth_deg = np.rad2deg(azimuth)
        polar_deg = np.rad2deg(polar)
        return (
            "\N{GREEK SMALL LETTER PHI}={:.2f}\N{GREEK SMALL LETTER PI} "
            "({:.2f}\N{DEGREE SIGN}), "
            "\N{GREEK SMALL LETTER RHO}={:.2f}\N{GREEK SMALL LETTER PI} "
            "({:.2f}\N{DEGREE SIGN})"
        ).format(azimuth / np.pi, azimuth_deg, polar / np.pi, polar_deg)

    def get_xaxis_transform(self, which="grid"):
        # Need to override this to get rid of spines.
        return self._xaxis_transform

    def get_yaxis_transform(self, which="grid"):
        # Need to override this to get rid of spines.
        return self._yaxis_transform

    def _gen_axes_spines(self):
        return {"stereographic": Spine.circular_spine(self, (0.5, 0.5), 0.5)}

    @staticmethod
    def _gen_axes_patch():
        return Circle((0.5, 0.5), 0.5)

    @staticmethod
    def get_data_ratio():
        return 1

    @staticmethod
    def can_pan():
        return False

    @staticmethod
    def can_zoom():
        return False

    def scatter(self, *args, **kwargs):
        new_kwargs = dict(zorder=3, clip_on=False)
        for k, v in new_kwargs.items():
            kwargs.setdefault(k, v)
        azimuth, polar = self._pretransform_input(args)
        super().scatter(azimuth, polar, **kwargs)

    def text(self, *args, **kwargs):
        new_kwargs = dict(va="bottom", ha="center", usetex=True)
        for k, v in new_kwargs.items():
            kwargs.setdefault(k, v)
        azimuth, polar = self._pretransform_input(args)
        super().text(azimuth, polar, **kwargs)

    # ----------- Custom attributes and methods below here ----------- #

    @property
    def pole(self):
        return self._pole

    @pole.setter
    def pole(self, value):
        if value not in [1, -1]:
            raise ValueError(f"Pole must be -1 (upper) or 1 (lower), not {value}")
        self._pole = value
        self._set_lim_and_transforms()
        self.clear()

    @property
    def hemisphere(self):
        return {"-1": "upper", "1": "lower"}[str(self.pole)]

    def show_hemisphere(self, **kwargs):
        new_kwargs = dict(ha="right", va="bottom")
        new_kwargs.update(kwargs)
        Axes.text(self, (3 / 4) * np.pi, np.pi / 2, s=self.hemisphere, **new_kwargs)

    def set_polar_grid(self, resolution=None):
        if resolution is not None:
            self._polar_resolution = resolution
        grid = np.arange(0, self._polar_cap, np.deg2rad(self._polar_resolution))
        self.set_yticks(grid)

    def set_azimuth_grid(self, resolution=None):
        if resolution is not None:
            self._azimuth_resolution = resolution
        grid = np.arange(0, self._azimuth_cap, np.deg2rad(self._azimuth_resolution))
        self.set_xticks(grid)

    def _set_label(self, x, y, label, **kwargs):
        bbox_dict = dict(boxstyle="round", fc="w", ec="w")
        new_kwargs = dict(ha="center", va="center", bbox=bbox_dict)
        new_kwargs.update(kwargs)
        super().text(x=x, y=y, s=label, **new_kwargs)

    def set_xlabel(self, label="X", **kwargs):
        self._set_label(0, self._polar_cap, label, **kwargs)

    def set_ylabel(self, label="Y", **kwargs):
        self._set_label(self._polar_cap, self._polar_cap, label, **kwargs)

    def set_zlabel(self, label="Z", **kwargs):
        self._set_label(0, 0, label, **kwargs)

    @staticmethod
    def _pretransform_input(values):
        """Return azimuth and polar angles from input data."""
        if len(values) == 2:
            azimuth, polar = values
        else:
            value = values[0]
            if isinstance(value, Rotation):
                v = value * Vector3d.zvector()
                azimuth = v.phi
                polar = v.theta
            elif isinstance(value, Vector3d):
                azimuth = value.phi
                polar = value.theta
            else:
                raise ValueError(
                    "Accepts only one (Vector3d or Rotation) or two (azimuth, "
                    "polar) input arguments"
                )
        return azimuth, polar


register_projection(StereographicPlot)
