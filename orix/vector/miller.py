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

from typing import Optional, Union

import numpy as np

from orix.vector import Vector3d


class Miller(Vector3d):
    """Miller indices, describing directions with respect to the crystal
    reference frame.
    """

    def __init__(
        self,
        xyz: Optional[Union[np.ndarray, list, tuple]] = None,
        uvw: Optional[Union[np.ndarray, list, tuple]] = None,
        hkl: Optional[Union[np.ndarray, list, tuple]] = None,
        hkil: Optional[Union[np.ndarray, list, tuple]] = None,
        phase: Optional = None,
    ):
        self.phase = phase
        self._print_format = "hkl"
        if xyz is not None:
            xyz = np.asarray(xyz)
            self._print_format = "xyz"
        elif uvw is not None:
            xyz = _uvw2xyz(uvw=uvw, lattice=phase.structure.lattice)
            self.print_format = "uvw"
        elif hkl is not None:
            xyz = _hkl2xyz(hkl=hkl, lattice=phase.structure.lattice)
        elif hkil is not None:
            hkil = np.asarray(hkil)
            _check_hkil(hkil)
            hkl = _hkil2hkl(hkil)
            xyz = _hkl2xyz(hkl=hkl, lattice=phase.structure.lattice)
            self.print_format = "hkil"
        else:
            raise ValueError(
                "Either `uvw` (direct), `hkl` (reciprocal), `hkil` (reciprocal), or "
                "`xyz` (assumes direct) coordinates must be passed"
            )
        super().__init__(xyz)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        shape = self.shape
        symmetry = None if self.phase is None else self.phase.point_group.name
        print_format = self.print_format
        if print_format == "xyz":
            data = self.data
        else:
            data = self.__getattribute__(print_format)
        data = np.array_str(data, precision=4, suppress_small=True)
        return f"{name} {shape}, point group {symmetry}, {print_format}\n" f"{data}"

    def __getitem__(self, key):
        return self.__class__(xyz=self.data[key], phase=self.phase)

    @property
    def print_format(self) -> str:
        return self._print_format

    @print_format.setter
    def print_format(self, value: str):
        value = value.lower()
        formats = ["xyz", "uvw", "hkl", "hkil"]
        if value not in formats:
            raise ValueError(f"Available print formats are {formats}")
        self._print_format = value

    @property
    def hkl(self) -> np.ndarray:
        return _xyz2hkl(xyz=self.data, lattice=self.phase.structure.lattice)

    @hkl.setter
    def hkl(self, value: np.ndarray):
        self.data = _hkl2xyz(hkl=value, lattice=self.phase.struture.lattice)

    @property
    def hkil(self) -> np.ndarray:
        return _hkl2hkil(self.hkl)

    @hkil.setter
    def hkil(self, value: np.ndarray):
        self.hkl = _hkil2hkl(value)

    @property
    def h(self) -> np.ndarray:
        return self.hkl[..., 0]

    @property
    def k(self) -> np.ndarray:
        return self.hkl[..., 1]

    @property
    def i(self) -> np.ndarray:
        return self.hkil[..., 2]

    @property
    def l(self) -> np.ndarray:
        return self.hkl[..., 2]

    @property
    def uvw(self) -> np.ndarray:
        return _xyz2uvw(xyz=self.data, lattice=self.phase.structure.lattice)

    @uvw.setter
    def uvw(self, value: np.ndarray):
        self.data = _uvw2xyz(uvw=value, lattice=self.phase.structure.lattice)

    @property
    def u(self) -> np.ndarray:
        return self.uvw[..., 0]

    @property
    def v(self) -> np.ndarray:
        return self.uvw[..., 1]

    @property
    def w(self) -> np.ndarray:
        return self.uvw[..., 2]

    @property
    def gspacing(self) -> np.ndarray:
        return self.phase.structure.lattice.rnorm(self.hkl)

    @property
    def dspacing(self) -> np.ndarray:
        return 1 / self.gspacing

    @property
    def multiplicity(self) -> np.ndarray:
        return super().symmetrise(
            symmetry=self.phase.point_group, unique=True, return_multiplicity=True,
        )[1]

    def symmetrise(self, unique: bool = False, return_index: bool = False):
        out = super().symmetrise(
            symmetry=self.phase.point_group,
            unique=unique,
            return_multiplicity=False,
            return_index=return_index,
        )
        if return_index:
            m, idx = out
            m.phase = self.phase
            return m, idx
        else:
            out.phase = self.phase
            return out

    def unique(self, return_index: bool = False, return_inverse: bool = False):
        out = super().unique(return_index=return_index, return_inverse=return_inverse)
        if return_index and return_inverse:
            m, idx, inv = out
            m.phase = self.phase
            return m, idx, inv
        elif return_index and not return_inverse:
            m, idx = out
            m.phase = self.phase
            return m, idx
        elif not return_index and return_inverse:
            m, inv = out
            m.phase = self.phase
            return m, inv
        else:
            out.phase = self.phase
            return out

    @property
    def allowed(self) -> np.ndarray:
        """Return whether planes diffract according to structure factor
        selection rules, assuming kinematical scattering theory.
        """
        self._raise_if_no_space_group()

        # Translational symmetry
        centering = self.phase.space_group.short_name[0]

        if centering == "P":  # Primitive
            if self.phase.space_group.crystal_system == "HEXAGONAL":
                # TODO: See rules in e.g.
                #  https://mcl1.ncifcrf.gov/dauter_pubs/284.pdf, Table 4
                #  http://xrayweb.chem.ou.edu/notes/symmetry.html, Systematic Absences
                raise NotImplementedError
            else:  # Any hkl
                return np.ones(self.size, dtype=bool)
        elif centering == "F":  # Face-centred, hkl all odd/even
            selection = np.sum(np.mod(self.hkl, 2), axis=1)
            return np.array([i not in [1, 2] for i in selection], dtype=bool)
        elif centering == "I":  # Body-centred, h + k + l = 2n (even)
            return np.mod(np.sum(self.hkl, axis=1), 2) == 0
        elif centering == "A":  # Centred on A faces only
            return np.mod(self.k + self.l, 2) == 0
        elif centering == "B":  # Centred on B faces only
            return np.mod(self.h + self.l, 2) == 0
        elif centering == "C":  # Centred on C faces only
            return np.mod(self.h + self.k, 2) == 0
        elif centering in ["R", "H"]:  # Rhombohedral
            return np.mod(-self.h + self.k + self.l, 3) == 0

    def _raise_if_no_space_group(self):
        """Raise ValueError if the phase attribute has no space group
        set.
        """
        if self.phase.space_group is None:
            raise ValueError(f"The phase {self.phase} must have a space group set")


def _uvw2xyz(uvw: Union[np.ndarray, list, tuple], lattice) -> np.ndarray:
    return np.asarray(uvw).dot(lattice.base)


def _xyz2uvw(xyz: Union[np.ndarray, list, tuple], lattice) -> np.ndarray:
    return np.linalg.inv(lattice.base.T).dot(xyz.T).T


def _hkl2xyz(hkl: Union[np.ndarray, list, tuple], lattice) -> np.ndarray:
    return np.asarray(hkl).dot(lattice.recbase.T)


def _xyz2hkl(xyz: np.ndarray, lattice) -> np.ndarray:
    xyz = np.asarray(xyz)
    return np.linalg.inv(lattice.recbase).dot(xyz.T).T


def _hkl2hkil(hkl: Union[np.ndarray, list, tuple]) -> np.ndarray:
    hkl = np.asarray(hkl)
    hkil = np.zeros(hkl.shape[:-1] + (4,))
    h = hkl[..., 0]
    k = hkl[..., 1]
    hkil[..., 0] = h
    hkil[..., 1] = k
    hkil[..., 2] = -(h + k)
    hkil[..., 3] = hkl[..., 2]
    return hkil


def _hkil2hkl(hkil: Union[np.ndarray, list, tuple]) -> np.ndarray:
    hkil = np.asarray(hkil)
    hkl = np.zeros(hkil.shape[:-1] + (3,))
    hkl[..., :2] = hkil[..., :2]
    hkl[..., 2] = hkil[..., 3]
    return hkl


def _check_hkil(hkil: Union[np.ndarray, list, tuple]):
    hkil = np.asarray(hkil)
    if not np.allclose(hkil[..., 0] + hkil[..., 1] + hkil[..., 2], 0, atol=1e-4):
        raise ValueError(
            "The Miller-Bravais indices convention h + k + i = 0 is not satisfied"
        )
