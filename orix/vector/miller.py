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

from itertools import product
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
        uvtw: Optional[Union[np.ndarray, list, tuple]] = None,
        hkl: Optional[Union[np.ndarray, list, tuple]] = None,
        hkil: Optional[Union[np.ndarray, list, tuple]] = None,
        phase=None,
        coordinate_format: str = None,
    ):
        self.phase = phase

        if xyz is not None:
            xyz = np.asarray(xyz)
            in_coords = "xyz"
        elif uvw is not None:
            xyz = _uvw2xyz(uvw=uvw, lattice=phase.structure.lattice)
            in_coords = "uvw"
        elif uvtw is not None:
            _check_uvtw(uvtw)
            uvw = _uvtw2uvw(uvtw=uvtw)
            xyz = _uvw2xyz(uvw=uvw, lattice=phase.structure.lattice)
            in_coords = "uvtw"
        elif hkl is not None:
            xyz = _hkl2xyz(hkl=hkl, lattice=phase.structure.lattice)
            in_coords = "hkl"
        elif hkil is not None:
            hkil = np.asarray(hkil)
            _check_hkil(hkil)
            hkl = _hkil2hkl(hkil)
            xyz = _hkl2xyz(hkl=hkl, lattice=phase.structure.lattice)
            in_coords = "hkil"
        else:
            raise ValueError(
                "Either `uvw` (direct), `uvtw` (direct), `hkl` (reciprocal), `hkil`"
                " (reciprocal), or `xyz` (assumes direct) coordinates must be passed"
            )
        super().__init__(xyz)

        if coordinate_format is None:
            coordinate_format = in_coords
        self.coordinate_format = coordinate_format

    def __repr__(self) -> str:
        name = self.__class__.__name__
        shape = self.shape
        symmetry = None if self.phase is None else self.phase.point_group.name
        coordinate_format = self.coordinate_format
        data = np.array_str(self._coordinates, precision=4, suppress_small=True)
        return (
            f"{name} {shape}, point group {symmetry}, {coordinate_format}\n" f"{data}"
        )

    def __getitem__(self, key):
        return self.__class__(
            xyz=self.data[key],
            phase=self.phase,
            coordinate_format=self.coordinate_format,
        )

    @property
    def coordinate_format(self) -> str:
        return self._coordinate_format

    @coordinate_format.setter
    def coordinate_format(self, value: str):
        value = value.lower()
        formats = ["xyz", "uvw", "uvtw", "hkl", "hkil"]
        if value not in formats:
            raise ValueError(f"Available print formats are {formats}")
        self._coordinate_format = value

    @property
    def _vector_format(self) -> str:
        coordinate_format = self.coordinate_format
        if coordinate_format == "xyz":
            return coordinate_format
        elif coordinate_format in ["hkl", "hkil"]:
            return "hkl"
        else:  # in ["uvw", "uvtw"]
            return "uvw"

    @property
    def _coordinates(self):
        coordinate_format = self.coordinate_format
        if coordinate_format == "xyz":
            coordinate_format = "data"
        return self.__getattribute__(coordinate_format)

    @property
    def _vector_coordinates(self):
        vector_format = self._vector_format
        if vector_format == "xyz":
            vector_format = "data"
        return self.__getattribute__(vector_format)

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
    def uvtw(self) -> np.ndarray:
        # U = 2u - v, V = 2v - u, T = -(u + v), W = 3w
        return _uvw2uvtw(self.uvw)

    @uvtw.setter
    def uvtw(self, value: np.ndarray):
        self.uvw = _uvtw2uvw(value)

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
    def U(self) -> np.ndarray:
        return self.uvtw[..., 0]

    @property
    def V(self) -> np.ndarray:
        return self.uvtw[..., 1]

    @property
    def T(self) -> np.ndarray:
        return self.uvtw[..., 2]

    @property
    def W(self) -> np.ndarray:
        return self.uvtw[..., 3]

    @property
    def gspacing(self) -> np.ndarray:
        return self.phase.structure.lattice.rnorm(self.hkl)

    @property
    def dspacing(self) -> np.ndarray:
        return 1 / self.gspacing

    @property
    def multiplicity(self) -> np.ndarray:
        return self.symmetrise(unique=True, return_multiplicity=True)[1]

    @property
    def _is_hexagonal(self) -> bool:
        return _is_hexagonal(self.phase.structure.lattice.abcABG()[3:])

    @classmethod
    def from_highest_indices(
        cls,
        phase,
        hkl: Optional[Union[np.ndarray, list, tuple]] = None,
        uvw: Optional[Union[np.ndarray, list, tuple]] = None,
    ):
        if hkl is not None:
            coordinate_format = "hkl"
            highest_idx = hkl
        elif uvw is not None:
            coordinate_format = "uvw"
            highest_idx = uvw
        else:
            raise ValueError("Either highest `hkl` or `uvw` indices must be passed")
        idx = _get_indices_from_highest(highest_indices=highest_idx)
        init_kw = {
            coordinate_format: idx,
            "phase": phase,
            "coordinate_format": coordinate_format,
        }
        return cls(**init_kw)

    @classmethod
    def from_min_dspacing(cls, phase, min_dspacing: float = 0.5):
        highest_hkl = _get_highest_hkl(
            lattice=phase.structure.lattice, min_dspacing=min_dspacing
        )
        hkl = _get_indices_from_highest(highest_indices=highest_hkl)
        return cls(hkl=hkl, phase=phase).unique()

    def cross(self, other):
        # TODO: Consider whether to use "zone axis" format instead
        return self.__class__(
            xyz=super().cross(other).data,
            phase=self.phase,
            coordinate_format=self.coordinate_format,
        )

    def symmetrise(
        self,
        unique: bool = False,
        return_multiplicity: bool = False,
        return_index: bool = False,
    ):
        if return_multiplicity and not unique:
            raise ValueError("`unique` must be True when `return_multiplicity` is True")
        elif return_index and not unique:
            raise ValueError("`unique` must be True when `return_index` is True")

        # Symmetrise directions with respect to crystal symmetry
        operations = self.phase.point_group
        v2 = operations.outer(self)

        if unique:
            n_v = self.size
            v3 = self.zero((n_v, operations.size))
            multiplicity = np.zeros(n_v, dtype=int)
            idx = np.ones(v3.size, dtype=int) * -1
            l_accum = 0
            for i in range(n_v):
                vi = v2[:, i].unique()
                l = vi.size
                v3[i, :l] = vi
                multiplicity[i] = l
                idx[l_accum : l_accum + l] = i
                l_accum += l
            non_zero = np.sum(np.abs(v3.data), axis=-1) != 0
            v2 = v3[non_zero]
            idx = idx[: np.sum(non_zero)]

        v2 = v2.flatten()

        # Carry over crystal structure and coordinate format
        m = self.__class__(
            xyz=v2.data, phase=self.phase, coordinate_format=self.coordinate_format
        )

        if return_multiplicity and return_index:
            return m, multiplicity, idx
        elif return_multiplicity and not return_index:
            return m, multiplicity
        elif not return_multiplicity and return_index:
            return m, idx
        else:
            return m

    def unique(self, use_symmetry: bool = False, return_index: bool = False):
        out = super().unique(return_index=return_index)
        if return_index:
            v, idx = out
        else:
            v = out

        if use_symmetry:
            operations = self.phase.point_group
            n_v = v.size
            v2 = operations.outer(v).flatten().reshape(*(n_v, operations.size))
            data = v2.data
            data_sorted = np.zeros_like(data)
            for i in range(n_v):
                a = data[i]
                order = np.lexsort(a.T)  # Sort by column 1, 2, then 3
                data_sorted[i] = a[order]
            _, idx = np.unique(data_sorted, return_index=True, axis=0)
            v = v[idx[::-1]]

        m = self.__class__(
            xyz=v.data, phase=self.phase, coordinate_format=self.coordinate_format,
        )
        if return_index:
            return m, idx
        else:
            return m


def _uvw2xyz(uvw, lattice):
    dsm = _direct_structure_matrix(lattice)
    return dsm.dot(np.asarray(uvw).T).T


def _xyz2uvw(xyz, lattice):
    rsm = _reciprocal_structure_matrix(lattice)
    return np.asarray(xyz).dot(rsm)


def _hkl2xyz(hkl, lattice):
    rsm = _reciprocal_structure_matrix(lattice)
    return rsm.dot(np.asarray(hkl).T).T


def _xyz2hkl(xyz, lattice):
    dsm = _direct_structure_matrix(lattice)
    return np.asarray(xyz).dot(dsm)


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
    if not np.allclose(np.sum(hkil[..., :3], axis=-1), 0, atol=1e-4):
        raise ValueError(
            "The Miller-Bravais indices convention h + k + i = 0 is not satisfied"
        )


def _uvw2uvtw(uvw: Union[np.ndarray, list, tuple]) -> np.ndarray:
    uvw = np.asarray(uvw)
    uvtw = np.zeros(uvw.shape[:-1] + (4,))
    u = uvw[..., 0]
    v = uvw[..., 1]

    # DeGraef: U = (2u - v) / 3, V = (2v - u) / 3, T = -(U + V), W = w
    #    big_u = ((2 * u) - v) / 3
    #    big_v = ((2 * v) - u) / 3
    #    uvtw[..., 0] = big_u
    #    uvtw[..., 1] = big_v
    #    uvtw[..., 2] = - (big_u + big_v)
    #    uvtw[..., 3] = uvw[..., 2]

    # MTEX: U = 2u - v, V = 2v - u, T = -(u + v), W = 3w
    uvtw[..., 0] = 2 * u - v
    uvtw[..., 1] = 2 * v - u
    uvtw[..., 2] = -(u + v)
    uvtw[..., 3] = 3 * uvw[..., 2]

    return uvtw


def _uvtw2uvw(uvtw: Union[np.ndarray, list, tuple]) -> np.ndarray:
    uvtw = np.asarray(uvtw)
    uvw = np.zeros(uvtw.shape[:-1] + (3,))

    # DeGraef: u = 2U + V, v = 2V + U, w = W
    #    big_u = uvtw[..., 0]
    #    big_v = uvtw[..., 1]
    #    uvw[..., 0] = 2 * big_u + big_v
    #    uvw[..., 1] = 2 * big_v + big_u
    #    uvw[..., 2] = uvtw[..., 3]

    # MTEX: u = 2U + V, v = 2V + U, w = W / 3
    big_u = uvtw[..., 0]
    big_v = uvtw[..., 1]
    big_t = uvtw[..., 2]
    uvw[..., 0] = big_u - big_t
    uvw[..., 1] = big_v - big_t
    uvw[..., 2] = uvtw[..., 3]
    uvw = uvw / 3

    return uvw


def _check_uvtw(uvtw: Union[np.ndarray, list, tuple]):
    uvtw = np.asarray(uvtw)
    if not np.allclose(np.sum(uvtw[..., :3], axis=-1), 0, atol=1e-4):
        raise ValueError(
            "The Miller-Bravais indices convention U + V + T = 0 is not satisfied"
        )


def _is_hexagonal(angles) -> bool:
    """Determine whether a lattice belongs to the hexagonal lattice
    family.
    """
    return np.allclose(angles, [90, 90, 120])


def _get_indices_from_highest(
    highest_indices: Union[np.ndarray, list, tuple]
) -> np.ndarray:
    """Return a list of coordinates from a set of highest indices.

    Parameters
    ----------
    highest_indices
        Highest indices to consider.

    Returns
    -------
    indices
        An array of indices sorted from positive to negative in the
        first column.
    """
    highest_indices = np.asarray(highest_indices)
    if not np.all(highest_indices >= 0) or np.all(highest_indices == 0):
        raise ValueError(
            f"All indices {highest_indices} must be positive with at least one non-zero"
        )
    index_ranges = [np.arange(-i, i + 1) for i in highest_indices]
    indices = np.asarray(list(product(*index_ranges)), dtype=int)
    indices = indices[~np.all(indices == 0, axis=1)]  # Remove (000)
    indices = indices[::-1]  # Make e.g. (111) first instead of (-1-1-1)
    return indices


def _get_highest_hkl(lattice, min_dspacing: Optional[float] = 0.5) -> np.ndarray:
    """Return the highest Miller indices hkl of the plane with a direct
    space interplanar spacing greater than but closest to a lower
    threshold.

    Parameters
    ----------
    lattice : diffpy.structure.Lattice
        Crystal lattice.
    min_dspacing
        Smallest interplanar spacing to consider. Default is 0.5 Å.

    Returns
    -------
    highest_hkl
        Highest Miller indices.
    """
    highest_hkl = np.ones(3, dtype=int)
    for i in range(3):
        hkl = np.zeros(3)
        d = min_dspacing + 1
        while d > min_dspacing:
            hkl[i] += 1
            d = 1 / lattice.rnorm(hkl)
        highest_hkl[i] = hkl[i]
    return highest_hkl


def _direct_structure_matrix(lattice):
    a, b, c = lattice.abcABG()[:3]
    ca, cb, cg = lattice.ca, lattice.cb, lattice.cg
    sg = lattice.sg
    return np.array(
        [
            [a, b * cg, c * cb],
            [0, b * sg, -c * (cb * cg - ca) / sg],
            [0, 0, lattice.volume / (a * b * sg)],
        ]
    )


def _reciprocal_structure_matrix(lattice):
    return np.linalg.inv(_direct_structure_matrix(lattice)).T
