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

from copy import deepcopy
from itertools import product
from typing import Optional, Union

import numpy as np

from orix.scalar import Scalar
from orix.vector import Vector3d


_PRECISION = np.finfo(np.float32).precision


class Miller(Vector3d):
    """Miller indices, describing directions with respect to the crystal
    reference frame.
    """

    def __init__(
        self,
        xyz: Optional[Union[np.ndarray, list, tuple]] = None,
        uvw: Optional[Union[np.ndarray, list, tuple]] = None,
        UVTW: Optional[Union[np.ndarray, list, tuple]] = None,
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
        elif UVTW is not None:
            _check_UVTW(UVTW)
            uvw = _UVTW2uvw(UVTW=UVTW)
            xyz = _uvw2xyz(uvw=uvw, lattice=phase.structure.lattice)
            in_coords = "UVTW"
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
                "Either `uvw` (direct), `UVTW` (direct), `hkl` (reciprocal), `hkil`"
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
        data = np.array_str(self.coordinates, precision=4, suppress_small=True)
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
        formats = ["xyz", "uvw", "UVTW", "hkl", "hkil"]
        if value not in formats:
            raise ValueError(f"Available print formats are {formats}")
        self._coordinate_format = value

    @property
    def coordinates(self):
        coordinate_format = self.coordinate_format
        if coordinate_format == "xyz":
            coordinate_format = "data"
        coordinates = self.__getattribute__(coordinate_format)
        return coordinates.round(decimals=_PRECISION)

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
    def UVTW(self) -> np.ndarray:
        return _uvw2UVTW(self.uvw)

    @UVTW.setter
    def UVTW(self, value: np.ndarray):
        self.uvw = _UVTW2uvw(value)

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
        return self.UVTW[..., 0]

    @property
    def V(self) -> np.ndarray:
        return self.UVTW[..., 1]

    @property
    def T(self) -> np.ndarray:
        return self.UVTW[..., 2]

    @property
    def W(self) -> np.ndarray:
        return self.UVTW[..., 3]

    @property
    def length(self) -> np.ndarray:
        if self.coordinate_format in ["hkl", "hkil"]:
            return self.phase.structure.lattice.rnorm(self.hkl)
        elif self.coordinate_format in ["uvw", "UVTW"]:
            return self.phase.structure.lattice.norm(self.uvw)
        else:
            return self.norm.data

    @property
    def multiplicity(self) -> np.ndarray:
        return self.symmetrise(unique=True, return_multiplicity=True)[1]

    @property
    def unit(self):
        return self.__class__(
            xyz=super().unit.data,
            phase=self.phase,
            coordinate_format=self.coordinate_format,
        )

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
        init_kw = {coordinate_format: idx, "phase": phase}
        return cls(**init_kw).unique()

    @classmethod
    def from_min_dspacing(cls, phase, min_dspacing: float = 0.5):
        highest_hkl = _get_highest_hkl(
            lattice=phase.structure.lattice, min_dspacing=min_dspacing
        )
        hkl = _get_indices_from_highest(highest_indices=highest_hkl)
        return cls(hkl=hkl, phase=phase).unique()

    def angle_with(self, other, use_symmetry: bool = False):
        if use_symmetry:
            other2 = other.symmetrise(unique=True)
            cosines = self.dot_outer(other2).data / (
                self.norm.data[..., np.newaxis] * other2.norm.data[np.newaxis, ...]
            )
            cosines = np.round(cosines, 9)
            angles = np.min(np.arccos(cosines), axis=-1)
            return Scalar(angles)
        else:
            return super().angle_with(other)

    def cross(self, other):
        new_fmt = dict(hkl="uvw", uvw="hkl", hkil="UVTW", UVTW="hkil")
        return self.__class__(
            xyz=super().cross(other).data,
            phase=self.phase,
            coordinate_format=new_fmt[self.coordinate_format],
        )

    def deepcopy(self):
        return deepcopy(self)

    def mean(self, use_symmetry: bool = False):
        if use_symmetry:
            return NotImplemented
        new_fmt = dict(hkl="uvw", uvw="hkl", hkil="UVTW", UVTW="hkil")
        return self.__class__(
            xyz=super().mean().data,
            phase=self.phase,
            coordinate_format=new_fmt[self.coordinate_format],
        )

    def round(self, max_index: int = 20):
        """Round a set of index triplet (Miller) or quartet
        (Miller-Bravais) to the *closest* smallest integers.

        Adopted from MTEX's Miller.round function.

        Parameters
        ----------
        indices
            Set of index triplet(s) or quartet(s) to round.
        max_index
            Maximum integer index to round to, by default 20.

        Return
        ------
        new_indices
            Integer array of rounded set of index triplet(s) or
            quartet(s).
        """
        if self.coordinate_format == "xyz":
            return self.deepcopy()
        else:
            new_coords = _round_indices(indices=self.coordinates, max_index=max_index)
            init_kw = {self.coordinate_format: new_coords, "phase": self.phase}
            return self.__class__(**init_kw)

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


def _uvw2UVTW(
    uvw: Union[np.ndarray, list, tuple], convention: Optional[str] = None
) -> np.ndarray:
    uvw = np.asarray(uvw)
    UVTW = np.zeros(uvw.shape[:-1] + (4,))
    u = uvw[..., 0]
    v = uvw[..., 1]
    # DeGraef: U = (2u - v) / 3, V = (2v - u) / 3, T = -(u + v) / 3, W = w
    UVTW[..., 0] = (2 * u - v) / 3
    UVTW[..., 1] = (2 * v - u) / 3
    UVTW[..., 2] = -(u + v) / 3
    UVTW[..., 3] = uvw[..., 2]
    if convention is not None and convention.lower() == "mtex":
        # MTEX: U = 2u - v, V = 2v - u, T = -(u + v), W = 3w
        UVTW *= 3
    return UVTW


def _UVTW2uvw(
    UVTW: Union[np.ndarray, list, tuple], convention: Optional[str] = None
) -> np.ndarray:
    UVTW = np.asarray(UVTW)
    uvw = np.zeros(UVTW.shape[:-1] + (3,))
    # DeGraef: u = 2U + V, v = 2V + U, w = W
    U = UVTW[..., 0]
    V = UVTW[..., 1]
    uvw[..., 0] = 2 * U + V
    uvw[..., 1] = U + 2 * V
    uvw[..., 2] = UVTW[..., 3]
    if convention is not None and convention.lower() == "mtex":
        # MTEX: u = 2U + V, v = 2V + U, w = W / 3
        uvw /= 3
    return uvw


def _check_UVTW(UVTW: Union[np.ndarray, list, tuple]):
    UVTW = np.asarray(UVTW)
    if not np.allclose(np.sum(UVTW[..., :3], axis=-1), 0, atol=1e-4):
        raise ValueError(
            "The Miller-Bravais indices convention U + V + T = 0 is not satisfied"
        )


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


# TODO: Implement in diffpy.structure.Lattice
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


# TODO: Implement in diffpy.structure.Lattice
def _reciprocal_structure_matrix(lattice):
    return np.linalg.inv(_direct_structure_matrix(lattice)).T


def _round_indices(
    indices: Union[np.ndarray, list, tuple], max_index: int = 12,
) -> np.ndarray:
    """Round a set of index triplet (Miller) or quartet (Miller-Bravais)
    to the *closest* smallest integers.

    Adopted from MTEX's Miller.round function.

    Parameters
    ----------
    indices
        Set of index triplet(s) or quartet(s) to round.
    max_index
        Maximum integer index to round to, by default 12.

    Return
    ------
    new_indices
        Integer array of rounded set of index triplet(s) or quartet(s).
    """
    # Allow list and tuple input (and don't overwrite `indices`)
    idx = np.asarray(indices)

    # Flatten and remove redundant third index if Miller-Bravais
    n_idx = idx.shape[-1]  # 3 or 4
    idx_flat = np.reshape(idx, (-1, n_idx))
    if n_idx == 4:
        idx_flat = idx_flat[:, [0, 1, 3]]

    # Get number of sets, max. index per set, and all possible integer
    # multipliers between 1 and `max_index`
    n_sets = idx_flat.size // 3
    max_per_set = np.max(np.abs(idx_flat), axis=-1)
    multipliers = np.arange(1, max_index + 1)

    # Divide by highest index, repeat array `max_index` number of times,
    # and multiply with all multipliers
    idx_scaled = (
        np.broadcast_to(idx_flat / max_per_set[:, np.newaxis], (max_index, n_sets, 3))
        * multipliers[:, np.newaxis, np.newaxis]
    )

    # Find the most suitable multiplier per set, which gives the
    # smallest error between the initial set and the scaled and rounded
    # set
    error = 1e-7 * np.round(
        1e7
        * np.sum((idx_scaled - np.round(idx_scaled)) ** 2, axis=-1)
        / np.sum(idx_scaled ** 2, axis=-1)
    )
    idx_min_error = np.argmin(error, axis=0)
    multiplier = (idx_min_error + 1) / max_per_set

    # Finally, multiply each set with their most suitable multiplier,
    # and round
    new_indices = np.round(multiplier[:, np.newaxis] * idx).astype(int)

    return new_indices
