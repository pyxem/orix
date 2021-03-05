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

from typing import Union

import numpy as np

from orix.vector import Vector3d


class Miller(Vector3d):
    def __init__(self, coordinates, phase=None):
        self.phase = phase
        super().__init__(coordinates)

    def __repr__(self) -> str:
        symmetry = None if self.phase is None else self.phase.point_group.name
        return (
            f"{self.__class__.__name__} {self.shape} {symmetry}\n"
            f"{np.array_str(self.data, precision=4, suppress_small=True)}"
        )

    def __getitem__(self, key):
        return self.__class__(coordinates=self.data[key], phase=self.phase)

    @property
    def hkl(self) -> np.ndarray:
        return self.data

    @hkl.setter
    def hkl(self, value: np.ndarray):
        self.data = value

    @property
    def h(self) -> np.ndarray:
        return self.hkl[..., 0]

    @property
    def k(self) -> np.ndarray:
        return self.hkl[..., 1]

    @property
    def l(self) -> np.ndarray:
        return self.hkl[..., 2]

    @property
    def hkil(self) -> np.ndarray:
        hkl = self.hkl
        # h k -(h+k) l
        return hkl

    @property
    def gspacing(self) -> np.ndarray:
        return self.phase.structure.lattice.rnorm(self.hkl)

    @property
    def dspacing(self) -> np.ndarray:
        return 1 / self.gspacing

    @property
    def multiplicity(self) -> Union[int, np.ndarray]:
        return self.symmetrise(antipodal=True, return_multiplicity=True)[1]

    def symmetrise(
        self,
        antipodal: bool = True,
        unique: bool = True,
        return_multiplicity: bool = False,
    ):
        """Return planes with symmetrically equivalent Miller indices.

        Parameters
        ----------
        antipodal : bool, optional
            Whether to include antipodal symmetry operations. Default
            is True.
        unique : bool, optional
            Whether to return only distinct indices. Default is True.
            If True, zero-entries, which are assumed to be degenerate,
            are removed.
        return_multiplicity : bool, optional
            Whether to return the multiplicity of indices. This option
            is only available if `unique` is True. Default is False.

        Returns
        -------
        ReciprocalLatticePoint
            Planes with Miller indices symmetrically equivalent to the
            original planes.
        multiplicity : numpy.ndarray
            Multiplicity of the original Miller indices. Only returned
            if `return_multiplicity` is True.
        """
        # Get symmetry operations
        pg = self.phase.point_group
        operations = pg if antipodal else pg[~pg.improper]

        out = get_equivalent_hkl(
            hkl=self.hkl,
            operations=operations,
            unique=unique,
            return_multiplicity=return_multiplicity,
        )

        # TODO: Enable inheriting classes pass on their properties in this new object
        # Format output and return
        if unique and return_multiplicity:
            multiplicity = out[1]
            if multiplicity.size == 1:
                multiplicity = multiplicity[0]
            return self.__class__(coordinates=out[0], phase=self.phase), multiplicity
        else:
            return self.__class__(coordinates=out, phase=self.phase)

    @property
    def allowed(self):
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


def get_equivalent_hkl(
    hkl, operations, unique: bool = False, return_multiplicity: bool = False
):
    """Return symmetrically equivalent Miller indices.

    Parameters
    ----------
    hkl : orix.vector.vector3d.Vector3d, numpy.ndarray, list or tuple\
            of int
        Miller indices.
    operations : orix.quaternion.symmetry.Symmetry
        Point group describing allowed symmetry operations.
    unique : bool, optional
        Whether to return only unique Miller indices. Default is False.
    return_multiplicity : bool, optional
        Whether to return the multiplicity of the input indices. Default
        is False.

    Returns
    -------
    new_hkl : orix.vector.Vector3d
        The symmetrically equivalent Miller indices.
    multiplicity : numpy.ndarray
        Number of symmetrically equivalent indices. Only returned if
        `return_multiplicity` is True.
    """
    new_hkl = operations.outer(Vector3d(hkl))
    new_hkl = new_hkl.flatten().reshape(*new_hkl.shape[::-1])

    multiplicity = None
    if unique:
        n_families = new_hkl.shape[0]
        multiplicity = np.zeros(n_families, dtype=int)
        temp_hkl = new_hkl[0].unique().data
        multiplicity[0] = temp_hkl.shape[0]
        if n_families > 1:
            for i, hkl in enumerate(new_hkl[1:]):
                temp_hkl2 = hkl.unique()
                multiplicity[i + 1] = temp_hkl2.size
                temp_hkl = np.append(temp_hkl, temp_hkl2.data, axis=0)
        new_hkl = Vector3d(temp_hkl[: multiplicity.sum()])

    # Remove 1-dimensions
    new_hkl = new_hkl.squeeze()

    if unique and return_multiplicity:
        return new_hkl, multiplicity
    else:
        return new_hkl
