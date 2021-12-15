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

import numpy as np

from orix.scalar import Scalar
from orix.vector import Vector3d


class Miller(Vector3d):
    """Direct crystal lattice vectors (uvw or UVTW) and reciprocal
    crystal lattice vectors (hkl or hkil), the latter known as Miller
    indices, describing directions with respect to the crystal reference
    frame defined by a phase's crystal lattice and symmetry.
    """

    def __init__(self, xyz=None, uvw=None, UVTW=None, hkl=None, hkil=None, phase=None):
        r"""Create a set of direct lattice vectors (uvw or UVTW)
        or reciprocal lattice vectors (hkl or hkil) describing
        directions with respect to a crystal reference frame defined by
        a phase's crystal lattice and symmetry.

        Exactly one of `xyz`, `uvw`, `UVTW`, `hkl`, or `hkil` must be
        passed.

        The vectors are stored internally as cartesian coordinates in
        the `data` attribute.

        Parameters
        ----------
        xyz : numpy.ndarray, list, or tuple, optional
            Vector(s) given in cartesian coordinates. Default is None.
        uvw : numpy.ndarray, list, or tuple, optional
            Indices of direct lattice vector(s). Default is None.
        UVTW : numpy.ndarray, list, or tuple, optional
            Indices of direct lattice vector(s), often preferred over
            `uvw` in trigonal and hexagonal lattices. Default is None.
        hkl : numpy.ndarray, list, or tuple, optional
            Indices of reciprocal lattice vector(s). Default is None.
        hkil : numpy.ndarray, list, or tuple, optional
            Indices of reciprocal lattice vector(s), often preferred
            over `hkl` in trigonal and hexagonal lattices. Default is
            None.
        phase : orix.crystal_map.Phase, optional
            A phase with a crystal lattice and symmetry. Must be passed
            whenever direct or reciprocal lattice vectors are created.

        Notes
        -----
        The Miller-Bravais indices :math:`UVTW` are defined as

        .. math::
            U &= \frac{2u - v}{3}, \\
            V &= \frac{2v - u}{3}, \\
            T &= -\frac{u + v}{3}, \\
            W &= w.
        """
        n_passed = np.sum([i is not None for i in [xyz, uvw, UVTW, hkl, hkil]])
        if n_passed == 0 or n_passed > 1:
            raise ValueError(
                "Exactly one of `xyz`, `uvw`, `UVTW`, `hkl`, `hkil` must be passed"
            )
        if xyz is None and phase is None:
            raise ValueError(
                "A phase with a crystal lattice and symmetry must be passed to create "
                "direct or reciprocal lattice vector(s)"
            )

        self.phase = phase
        if xyz is not None:
            xyz = np.asarray(xyz)
            self.coordinate_format = "xyz"
        elif uvw is not None:
            xyz = _uvw2xyz(uvw=uvw, lattice=phase.structure.lattice)
            self.coordinate_format = "uvw"
        elif UVTW is not None:
            _check_UVTW(UVTW)
            uvw = _UVTW2uvw(UVTW=UVTW)
            xyz = _uvw2xyz(uvw=uvw, lattice=phase.structure.lattice)
            self.coordinate_format = "UVTW"
        elif hkl is not None:
            xyz = _hkl2xyz(hkl=hkl, lattice=phase.structure.lattice)
            self.coordinate_format = "hkl"
        elif hkil is not None:
            hkil = np.asarray(hkil)
            _check_hkil(hkil)
            hkl = _hkil2hkl(hkil)
            xyz = _hkl2xyz(hkl=hkl, lattice=phase.structure.lattice)
            self.coordinate_format = "hkil"
        super().__init__(xyz)

    def __repr__(self):
        """String representation."""
        name = self.__class__.__name__
        shape = self.shape
        symmetry = None if self.phase is None else self.phase.point_group.name
        coordinate_format = self.coordinate_format
        data = np.array_str(self.coordinates, precision=4, suppress_small=True)
        return (
            f"{name} {shape}, point group {symmetry}, {coordinate_format}\n" f"{data}"
        )

    def __getitem__(self, key):
        """NumPy fancy indexing of vectors."""
        m = self.__class__(xyz=self.data[key], phase=self.phase).deepcopy()
        m.coordinate_format = self.coordinate_format
        return m

    # ---------------------- Unique properties ---------------------- #

    @property
    def coordinate_format(self):
        """Vector coordinate format, either "xyz", "uvw", "UVTW", "hkl",
        or "hkil".
        """
        return self._coordinate_format

    @coordinate_format.setter
    def coordinate_format(self, value):
        """Set the vector coordinate format, either "xyz", "uvw",
        "UVTW", "hkl", or "hkil".
        """
        formats = ["xyz", "uvw", "UVTW", "hkl", "hkil"]
        if value not in formats:
            raise ValueError(f"Available coordinate formats are {formats}")
        self._coordinate_format = value

    @property
    def coordinates(self):
        """Vector coordinates."""
        coordinate_format = self.coordinate_format
        if coordinate_format == "xyz":
            coordinate_format = "data"
        coordinates = self.__getattribute__(coordinate_format)
        return coordinates

    @property
    def hkl(self):
        """Reciprocal lattice vectors."""
        return _xyz2hkl(xyz=self.data, lattice=self.phase.structure.lattice)

    @hkl.setter
    def hkl(self, value):
        """Set the reciprocal lattice vectors."""
        self.data = _hkl2xyz(hkl=value, lattice=self.phase.structure.lattice)

    @property
    def hkil(self):
        """Reciprocal lattice vectors expressed as 4-index
        Miller-Bravais indices.
        """
        return _hkl2hkil(self.hkl)

    @hkil.setter
    def hkil(self, value):
        r"""Set the reciprocal lattice vectors expressed as 4-index
        Miller-Bravais indices. The sum of the first three indices,
        :math:`h`, :math:`k`, and :math:`i` must be zero.
        """
        self.hkl = _hkil2hkl(value)

    @property
    def h(self):
        """First reciprocal lattice vector index."""
        return self.hkl[..., 0]

    @property
    def k(self):
        """Second reciprocal lattice vector index."""
        return self.hkl[..., 1]

    @property
    def i(self):
        r"""Third reciprocal lattice vector index in 4-index
        Miller-Bravais indices, equal to :math:`-(h + k)`.
        """
        return self.hkil[..., 2]

    @property
    def l(self):
        """Third reciprocal lattice vector index, or fourth index in
        4-index Miller Bravais indices.
        """
        return self.hkl[..., 2]

    @property
    def uvw(self):
        """Direct lattice vectors."""
        return _xyz2uvw(xyz=self.data, lattice=self.phase.structure.lattice)

    @uvw.setter
    def uvw(self, value):
        """Set the direct lattice vectors."""
        self.data = _uvw2xyz(uvw=value, lattice=self.phase.structure.lattice)

    @property
    def UVTW(self):
        r"""Direct lattice vectors expressed as 4-index Weber symbols.
        They are defined as

        .. math::
            U &= \frac{2u - v}{3}, \\
            V &= \frac{2v - u}{3}, \\
            T &= -\frac{u + v}{3}, \\
            W &= w.
        """
        return _uvw2UVTW(self.uvw)

    @UVTW.setter
    def UVTW(self, value):
        r"""Set the direct lattice vectors expressed as 4-index Weber
        symbols. The sum of the first three indices, :math:`U`,
        :math:`V`, and :math:`T` must be zero.
        """
        self.uvw = _UVTW2uvw(value)

    @property
    def u(self):
        """First direct lattice vector index."""
        return self.uvw[..., 0]

    @property
    def v(self):
        """Second direct lattice vector index."""
        return self.uvw[..., 1]

    @property
    def w(self):
        """Third direct lattice vector index."""
        return self.uvw[..., 2]

    @property
    def U(self):
        r"""First direct lattice vector index in 4-index Weber symbols,
        equal to :math:`(2u - v)/(3)`.
        """
        return self.UVTW[..., 0]

    @property
    def V(self):
        r"""Second direct lattice vector index in 4-index Weber symbols,
        equal to :math:`(2v - u)/3`.
        """
        return self.UVTW[..., 1]

    @property
    def T(self):
        r"""Third direct lattice vector index in 4-index Weber symbols,
        equal to :math:`-(u + v)/3`.
        """
        return self.UVTW[..., 2]

    @property
    def W(self):
        r"""Fourth direct lattice vector index in 4-index Weber symbols,
        equal to :math:`w`.
        """
        return self.UVTW[..., 3]

    @property
    def length(self):
        """Length of each vector given in lattice parameter units if
        the :attr:`coordinate_format` attribute equals "uvw" or "UVTW",
        and inverse lattice parameter units if the attribute equals
        "hkl" or "hkil". If the attribute equals "xyz", the norms of the
        vectors in :attr:`data` are returned.
        """
        if self.coordinate_format in ["hkl", "hkil"]:
            return self.phase.structure.lattice.rnorm(self.hkl)
        elif self.coordinate_format in ["uvw", "UVTW"]:
            return self.phase.structure.lattice.norm(self.uvw)
        else:
            return self.norm.data

    @property
    def multiplicity(self):
        """Number of symmetrically equivalent directions per vector."""
        l = self.symmetrise(unique=True, return_multiplicity=True)[1]
        return l.reshape(self.shape)

    @property
    def space(self):
        """Whether the vector is in "direct" or "reciprocal" space."""
        return dict(
            xyz="direct",
            uvw="direct",
            UVTW="direct",
            hkl="reciprocal",
            hkil="reciprocal",
        )[self.coordinate_format]

    @property
    def is_hexagonal(self):
        """Whether the crystal reference frame is hexagonal/trigonal or
        not.
        """
        return self.phase.is_hexagonal

    # ----------- Overwritten Vector3d/Object3d properties ----------- #

    @property
    def unit(self):
        """Unit vectors."""
        m = self.__class__(xyz=super().unit.data, phase=self.phase)
        m.coordinate_format = self.coordinate_format
        return m

    # ------------------------ Unique methods ------------------------ #

    @classmethod
    def from_highest_indices(cls, phase, uvw=None, hkl=None):
        """Create a set of unique direct or reciprocal lattice vectors
        from three highest indices and a phase (crystal lattice and
        symmetry).

        Parameters
        ----------
        phase : orix.crystal_map.Phase
            A phase with a crystal lattice and symmetry.
        uvw : numpy.ndarray, list, or tuple, optional
            Three highest direct lattice vector indices.
        hkl : numpy.ndarray, list, or tuple, optional
            Three highest reciprocal lattice vector indices.
        """
        if uvw is not None:
            coordinate_format = "uvw"
            highest_idx = uvw
        elif hkl is not None:
            coordinate_format = "hkl"
            highest_idx = hkl
        else:
            raise ValueError("Either highest `hkl` or `uvw` indices must be passed")
        idx = _get_indices_from_highest(highest_indices=highest_idx)
        init_kw = {coordinate_format: idx, "phase": phase}
        return cls(**init_kw).unique()

    @classmethod
    def from_min_dspacing(cls, phase, min_dspacing=0.05):
        """Create a set of unique reciprocal lattice vectors with a
        a direct space interplanar spacing greater than a lower
        threshold.

        Parameters
        ----------
        phase : orix.crystal_map.Phase
            A phase with a crystal lattice and symmetry.
        min_dspacing : float, optional
            Smallest interplanar spacing to consider. Default is 0.05,
            in the unit used to define the lattice parameters in
            `phase`.
        """
        highest_hkl = _get_highest_hkl(
            lattice=phase.structure.lattice, min_dspacing=min_dspacing
        )
        hkl = _get_indices_from_highest(highest_indices=highest_hkl)
        return cls(hkl=hkl, phase=phase).unique()

    def deepcopy(self):
        """Return a deepcopy of the instance."""
        return deepcopy(self)

    def round(self, max_index=20):
        """Round a set of index triplet (Miller) or quartet
        (Miller-Bravais/Weber) to the *closest* smallest integers.

        Adopted from MTEX's Miller.round function.

        Parameters
        ----------
        max_index : int
            Maximum integer index to round to, by default 20.

        Return
        ------
        Miller
            Rounded set of index triplet(s) or quartet(s).
        """
        if self.coordinate_format == "xyz":
            return self.deepcopy()
        else:
            new_coords = _round_indices(indices=self.coordinates, max_index=max_index)
            init_kw = {self.coordinate_format: new_coords, "phase": self.phase}
            return self.__class__(**init_kw)

    def symmetrise(self, unique=False, return_multiplicity=False, return_index=False):
        """Vectors symmetrically equivalent to the ones in `self`.

        Parameters
        ----------
        unique : bool, optional
            Whether to return only unique vectors. Default is False.
        return_multiplicity : bool, optional
            Whether to return the multiplicity of each vector. Default
            is False.
        return_index : bool, optional
            Whether to return the index into `self` for the returned
            symmetrically equivalent vectors. Default is False.

        Returns
        -------
        Miller
            Flattened symmetrically equivalent vectors.
        multiplicity : numpy.ndarray
            Multiplicity of each vector. Returned if
            `return_multiplicity` is True.
        idx : numpy.ndarray
            Index into `self` for the symmetrically equivalent vectors.
            Returned if `return_index` is True.
        """
        if return_multiplicity and not unique:
            raise ValueError("`unique` must be True when `return_multiplicity` is True")
        elif return_index and not unique:
            raise ValueError("`unique` must be True when `return_index` is True")

        # Symmetrise directions with respect to crystal symmetry on the
        # flattened set of vectors
        operations = self.phase.point_group
        v2 = operations.outer(self.flatten())

        if unique:
            n_v = self.size  # Number of initial vectors in `self`

            # Array for symmetrically equivalent vectors
            v3 = self.zero((n_v, operations.size))

            # Array for multiplicity of initial vectors
            multiplicity = np.zeros(n_v, dtype=int)

            # Array for index into `self` for the returned symmetrically
            # equivalent vectors
            idx = np.ones(v3.size, dtype=int) * -1

            # Loop over initial vectors
            l_accum = 0
            for i in range(n_v):
                # Unique vectors among those symmetrically equivalent
                vi = v2[:, i].unique()
                l = vi.size  # Multiplicity
                v3[i, :l] = vi  # Insert only the unique ones

                # Multiplicity of this initial vector
                multiplicity[i] = l

                # Index into `self` for the unique, symmetrically
                # equivalent vectors
                idx[l_accum : l_accum + l] = i
                l_accum += l

            # Remove entries into `v3` and `idx` not used
            non_zero = np.sum(np.abs(v3.data), axis=-1) != 0
            v2 = v3[non_zero]
            idx = idx[: np.sum(non_zero)]

        v2 = v2.flatten()

        # Carry over crystal structure and coordinate format
        m = self.__class__(xyz=v2.data, phase=self.phase)
        m.coordinate_format = self.coordinate_format

        if return_multiplicity and return_index:
            return m, multiplicity, idx
        elif return_multiplicity and not return_index:
            return m, multiplicity
        elif not return_multiplicity and return_index:
            return m, idx
        else:
            return m

    def _compatible_with(self, other, raise_error=False):
        """Whether `self` and `other` have both the same crystal lattice
        and symmetry and that the vectors are in the same space.

        Parameters
        ----------
        other : Miller
        raise_error : bool, optional
            Whether to raise a ValueError if the instances are not
            compatible. Default is False.

        Returns
        -------
        bool
        """
        same_symmetry = self.phase.point_group.name == other.phase.point_group.name
        same_lattice = np.allclose(
            self.phase.structure.lattice.abcABG(),
            other.phase.structure.lattice.abcABG(),
        )
        same_space = self.space == other.space
        compatible = same_symmetry * same_lattice * same_space
        if not compatible and raise_error:
            raise ValueError(
                "The crystal lattices and symmetries must be the same, and the vectors "
                "must be in the same space"
            )
        else:
            return compatible

    # ------------- Overwritten Vector3d/Object3d methods- ----------- #

    def angle_with(self, other, use_symmetry=False):
        """Calculate angles between vectors in `self` and `other`,
        possibly using symmetrically equivalent vectors to find the
        smallest angle under symmetry.

        Vectors must have compatible shapes, and be in the same space
        (direct or recprocal) and crystal reference frames.

        Parameters
        ----------
        other : Vector3d or Miller
        use_symmetry : bool, optional
            Whether to consider equivalent vectors to find the smallest
            angle under symmetry. Default is False.

        Returns
        -------
        Scalar
            The angle between the vectors, in radians.
        """
        self._compatible_with(other, raise_error=True)
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
        """Cross product of a direct or reciprocal lattice vector with
        another vector, which is considered the zone axis between the
        vectors.

        Vectors must have compatible shapes, and be in the same space
        (direct or recprocal) and crystal reference frames.

        Returns
        -------
        Miller
            Vectors in reciprocal (direct) space if direct (reciprocal)
            vectors are crossed.
        """
        self._compatible_with(other, raise_error=True)
        new_fmt = dict(hkl="uvw", uvw="hkl", hkil="UVTW", UVTW="hkil")
        m = self.__class__(xyz=super().cross(other).data, phase=self.phase)
        m.coordinate_format = new_fmt[self.coordinate_format]
        return m

    def dot(self, other):
        """Dot product of a vector with another vector.

        Vectors must have compatible shapes, and be in the same space
        (direct or recprocal) and crystal reference frames.

        Returns
        -------
        Scalar
        """
        self._compatible_with(other, raise_error=True)
        return super().dot(other)

    def dot_outer(self, other):
        """Outer dot product of a vector with another vector.

        Vectors must be in the same space (direct or recprocal) and
        crystal reference frames.

        The dot product for every combination of vectors in `self` and
        `other` is computed.

        Returns
        -------
        Scalar
        """
        self._compatible_with(other, raise_error=True)
        return super().dot_outer(other)

    def flatten(self):
        m = self.__class__(xyz=super().flatten().data, phase=self.phase)
        m.coordinate_format = self.coordinate_format
        return m

    def transpose(self, *axes):
        """Returns a new Miller object containing the same data transposed.

        If ndim is originally 2, then order may be undefined.
        In this case the first two dimensions will be transposed.

        Parameters
        ----------
        axes : int, optional
            The transposed axes order. Only navigation axes need to be defined.
            May be undefined if self only contains two navigation dimensions.

        Returns
        -------
        Miller :
            A transposed Miller instance of the object.

        """
        m = self.__class__(xyz=super().transpose(*axes).data, phase=self.phase)
        m.coordinate_format = self.coordinate_format
        return m

    def get_nearest(self):
        """NotImplemented."""
        return NotImplemented

    def mean(self, use_symmetry=False):
        """Mean vector of the set of vectors."""
        # TODO: Allow using symmetry by projecting to fundamental sector
        if use_symmetry:
            return NotImplemented
        m = self.__class__(xyz=super().mean().data, phase=self.phase)
        m.coordinate_format = self.coordinate_format
        return m

    def reshape(self, *shape):
        m = self.__class__(xyz=super().reshape(*shape).data, phase=self.phase)
        m.coordinate_format = self.coordinate_format
        return m

    def unique(self, use_symmetry=False, return_index=False):
        """Unique vectors in `self`.

        Parameters
        ----------
        use_symmetry : bool, optional
            Whether to consider equivalent vectors to compute the unique
            vectors. Default is False.
        return_index : bool, optional
            Whether to return the indices of the (flattened) data where
            the unique entries were found. Default is False.

        Returns
        -------
        Miller
            Flattened unique vectors.
        idx : numpy.ndarray
            Indices of the unique data in the (flattened) array.
        """
        out = super().unique(return_index=return_index)
        if return_index:
            v, idx = out
        else:
            v = out

        if use_symmetry:
            operations = self.phase.point_group
            n_v = v.size
            v2 = operations.outer(v).flatten()
            v2 = operations.outer(v).flatten().reshape(*(n_v, operations.size))
            data = v2.data
            data_sorted = np.zeros_like(data)
            for i in range(n_v):
                a = data[i]
                order = np.lexsort(a.T)  # Sort by column 1, 2, then 3
                data_sorted[i] = a[order]
            _, idx = np.unique(data_sorted, return_index=True, axis=0)
            v = v[idx[::-1]]

        m = self.__class__(xyz=v.data, phase=self.phase)
        m.coordinate_format = self.coordinate_format
        if return_index:
            return m, idx
        else:
            return m


def _uvw2xyz(uvw, lattice):
    uvw = np.asarray(uvw)
    shape = uvw.shape
    uvw = uvw.reshape((uvw.size // 3, 3))
    xyz = _direct_structure_matrix(lattice).dot(uvw.T).T
    return xyz.reshape(shape)


def _xyz2uvw(xyz, lattice):
    xyz = np.asarray(xyz)
    shape = xyz.shape
    xyz = xyz.reshape((xyz.size // 3, 3))
    uvw = xyz.dot(_reciprocal_structure_matrix(lattice))
    return uvw.reshape(shape)


def _hkl2xyz(hkl, lattice):
    hkl = np.asarray(hkl)
    shape = hkl.shape
    hkl = hkl.reshape((hkl.size // 3, 3))
    xyz = _reciprocal_structure_matrix(lattice).dot(hkl.T).T
    return xyz.reshape(shape)


def _xyz2hkl(xyz, lattice):
    xyz = np.asarray(xyz)
    shape = xyz.shape
    xyz = xyz.reshape((xyz.size // 3, 3))
    hkl = xyz.dot(_direct_structure_matrix(lattice))
    return hkl.reshape(shape)


def _hkl2hkil(hkl):
    hkl = np.asarray(hkl)
    hkil = np.zeros(hkl.shape[:-1] + (4,))
    h = hkl[..., 0]
    k = hkl[..., 1]
    hkil[..., 0] = h
    hkil[..., 1] = k
    hkil[..., 2] = -(h + k)
    hkil[..., 3] = hkl[..., 2]
    return hkil


def _hkil2hkl(hkil):
    hkil = np.asarray(hkil)
    hkl = np.zeros(hkil.shape[:-1] + (3,))
    hkl[..., :2] = hkil[..., :2]
    hkl[..., 2] = hkil[..., 3]
    return hkl


def _check_hkil(hkil):
    hkil = np.asarray(hkil)
    if not np.allclose(np.sum(hkil[..., :3], axis=-1), 0, atol=1e-4):
        raise ValueError(
            "The Miller-Bravais indices convention h + k + i = 0 is not satisfied"
        )


def _uvw2UVTW(uvw, convention=None):
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


def _UVTW2uvw(UVTW, convention=None):
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


def _check_UVTW(UVTW):
    UVTW = np.asarray(UVTW)
    if not np.allclose(np.sum(UVTW[..., :3], axis=-1), 0, atol=1e-4):
        raise ValueError(
            "The Miller-Bravais indices convention U + V + T = 0 is not satisfied"
        )


def _get_indices_from_highest(highest_indices):
    """Return a list of coordinates from a set of highest indices.

    Parameters
    ----------
    highest_indices : list, tuple, or numpy.ndarray
        Highest indices to consider.

    Returns
    -------
    indices : numpy.ndarray
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


def _get_highest_hkl(lattice, min_dspacing=0.05):
    """Return the highest Miller indices hkl of the plane with a direct
    space interplanar spacing closest to a threshold.

    Parameters
    ----------
    lattice : diffpy.structure.Lattice
        Crystal lattice.
    min_dspacing : float, optional
        Smallest interplanar spacing to consider. Default is 0.05.

    Returns
    -------
    highest_hkl : numpy.ndarray
        Highest Miller indices.
    """
    highest_hkl = np.ones(3, dtype=int)
    for i in range(3):
        hkl = np.zeros(3)
        hkl[i] = 1
        d = 1 / lattice.rnorm(hkl)
        while d > min_dspacing:
            hkl[i] += 1
            d = 1 / lattice.rnorm(hkl)
        highest_hkl[i] = hkl[i] - 1
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


def _round_indices(indices, max_index=12):
    """Round a set of index triplet (Miller) or quartet (Miller-Bravais)
    to the *closest* smallest integers.

    Adopted from MTEX's Miller.round function.

    Parameters
    ----------
    indices : list, tuple, or numpy.ndarray
        Set of index triplet(s) or quartet(s) to round.
    max_index : int, optional
        Maximum integer index to round to, by default 12.

    Return
    ------
    new_indices : numpy.ndarray
        Integer array of rounded set of index triplet(s) or quartet(s).
    """
    # Allow list and tuple input (and don't overwrite `indices`)
    idx = np.asarray(indices)

    # Flatten and remove redundant third index if Miller-Bravais
    n_idx = idx.shape[-1]  # 3 or 4
    idx_flat = np.reshape(idx, (-1, n_idx))
    if n_idx == 4:
        idx_flat = idx_flat[..., [0, 1, 3]]

    # Get number of sets, max. index per set, and all possible integer
    # multipliers between 1 and `max_index`
    n_sets = idx_flat.size // 3
    max_per_set = np.max(np.abs(idx_flat), axis=-1)
    multipliers = np.arange(1, max_index + 1)

    # Divide by highest index, repeat array `max_index` number of times,
    # and multiply with all multipliers
    idx_scaled = (
        np.broadcast_to(idx_flat / max_per_set[..., np.newaxis], (max_index, n_sets, 3))
        * multipliers[..., np.newaxis, np.newaxis]
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

    # Reshape `multiplier` to match indices shape
    multiplier = multiplier.reshape(idx.shape[:-1])[..., np.newaxis]

    # Finally, multiply each set with their most suitable multiplier,
    # and round
    new_indices = np.round(multiplier * idx).astype(int)

    return new_indices
