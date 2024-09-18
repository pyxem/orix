# Copyright 2018-2024 the orix developers
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

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING, Optional, Tuple, Union

try:
    # New in Python 3.11
    from typing import Self
except ImportError:  # pragma: no cover
    from typing_extensions import Self

from diffpy.structure import Lattice
import numpy as np

from orix.vector import Vector3d

if TYPE_CHECKING:  # pragma: no cover
    from orix.crystal_map import Phase
    from orix.quaternion import Symmetry


class Miller(Vector3d):
    r"""Direct crystal lattice vectors (uvw or UVTW) and reciprocal
    crystal lattice vectors (hkl or hkil), the latter known as Miller
    indices, describing directions with respect to the crystal reference
    frame defined by a phase's crystal lattice and symmetry.

    Exactly one of ``xyz``, ``uvw``, ``UVTW``, ``hkl``, or ``hkil``
    must be passed.

    The vectors are stored internally as cartesian coordinates in
    :attr:`data`.

    Parameters
    ----------
    xyz
        Vector(s) given in cartesian coordinates. Default is ``None``.
    uvw
        Indices of direct lattice vector(s). Default is ``None``.
    UVTW
        Indices of direct lattice vector(s), often preferred over
        *uvw* in trigonal and hexagonal lattices. Default is ``None``.
    hkl
        Indices of reciprocal lattice vector(s). Default is ``None``.
    hkil
        Indices of reciprocal lattice vector(s), often preferred over
        *hkl* in trigonal and hexagonal lattices. Default is ``None``.
    phase
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

    def __init__(
        self,
        xyz: Union[np.ndarray, list, tuple, None] = None,
        uvw: Union[np.ndarray, list, tuple, None] = None,
        UVTW: Union[np.ndarray, list, tuple, None] = None,
        hkl: Union[np.ndarray, list, tuple, None] = None,
        hkil: Union[np.ndarray, list, tuple, None] = None,
        phase: Optional["Phase"] = None,
    ) -> None:
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
            xyz = _transform_space(uvw, "d", "c", phase.structure.lattice)
            self.coordinate_format = "uvw"
        elif UVTW is not None:
            UVTW = np.asarray(UVTW)
            _check_UVTW(UVTW)
            uvw = _UVTW2uvw(UVTW=UVTW)
            xyz = _transform_space(uvw, "d", "c", phase.structure.lattice)
            self.coordinate_format = "UVTW"
        elif hkl is not None:
            xyz = _transform_space(hkl, "r", "c", phase.structure.lattice)
            self.coordinate_format = "hkl"
        elif hkil is not None:
            hkil = np.asarray(hkil)
            _check_hkil(hkil)
            hkl = _hkil2hkl(hkil)
            xyz = _transform_space(hkl, "r", "c", phase.structure.lattice)
            self.coordinate_format = "hkil"
        super().__init__(xyz)

    # -------------------------- Properties -------------------------- #

    @property
    def coordinate_format(self) -> str:
        """Return or set the vector coordinate format.

        Parameters
        ----------
        value : str
            Vector coordinate format, either ``"xyz"``, ``"uvw"``,
            ``"UVTW"``, ``"hkl"``, or ``"hkil"``.
        """
        return self._coordinate_format

    @coordinate_format.setter
    def coordinate_format(self, value: str) -> None:
        """Set the vector coordinate format."""
        formats = ["xyz", "uvw", "UVTW", "hkl", "hkil"]
        if value not in formats:
            raise ValueError(f"Available coordinate formats are {formats}")
        self._coordinate_format = value

    @property
    def coordinates(self) -> np.ndarray:
        """Return the vector coordinates."""
        coordinate_format = self.coordinate_format
        if coordinate_format == "xyz":
            coordinate_format = "data"
        coordinates = self.__getattribute__(coordinate_format)
        return coordinates

    @property
    def hkl(self) -> np.ndarray:
        """Return or set the reciprocal lattice vectors.

        Parameters
        ----------
        value : np.ndarray
            New reciprocal lattice vector array.
        """
        return _transform_space(self.data, "c", "r", self.phase.structure.lattice)

    @hkl.setter
    def hkl(self, value: np.ndarray) -> None:
        """Set the reciprocal lattice vectors."""
        self.data = _transform_space(value, "r", "c", self.phase.structure.lattice)

    @property
    def hkil(self) -> np.ndarray:
        r"""Return or set the reciprocal lattice vectors expressed as
        4-index Miller-Bravais indices.

        Parameters
        ----------
        value : np.ndarray
            New reciprocal lattice vector array. The sum of the first
            three indices, :math:`h`, :math:`k`, and :math:`i` must be
            zero.
        """
        return _hkl2hkil(self.hkl)

    @hkil.setter
    def hkil(self, value: np.ndarray) -> None:
        """Set the reciprocal lattice vectors expressed as 4-index
        Miller-Bravais indices.
        """
        self.hkl = _hkil2hkl(value)

    @property
    def h(self) -> np.ndarray:
        """Return the first reciprocal lattice vector index."""
        return self.hkl[..., 0]

    @property
    def k(self) -> np.ndarray:
        """Return the second reciprocal lattice vector index."""
        return self.hkl[..., 1]

    @property
    def i(self) -> np.ndarray:
        r"""Return the third reciprocal lattice vector index in 4-index
        Miller-Bravais indices, equal to :math:`-(h + k)`.
        """
        return self.hkil[..., 2]

    @property
    def l(self) -> np.ndarray:
        """Return the third reciprocal lattice vector index, or fourth
        index in 4-index Miller Bravais indices.
        """
        return self.hkl[..., 2]

    @property
    def uvw(self) -> np.ndarray:
        """Return or set the direct lattice vectors.

        Parameters
        ----------
        value : np.ndarray
            New direct lattice vector array.
        """
        return _transform_space(self.data, "c", "d", self.phase.structure.lattice)

    @uvw.setter
    def uvw(self, value: np.ndarray) -> None:
        """Set the direct lattice vectors."""
        self.data = _transform_space(value, "d", "c", self.phase.structure.lattice)

    @property
    def UVTW(self) -> np.ndarray:
        r"""Return or set the direct lattice vectors expressed as
        4-index Weber symbols.

        They are defined as

        .. math::
            U &= \frac{2u - v}{3}, \\
            V &= \frac{2v - u}{3}, \\
            T &= -\frac{u + v}{3}, \\
            W &= w.

        Parameters
        ----------
        value : np.ndarray
            New direct lattice vector array. The sum of the first three
            indices, :math:`U`, :math:`V`, and :math:`T` must be zero.
        """
        return _uvw2UVTW(self.uvw)

    @UVTW.setter
    def UVTW(self, value: np.ndarray) -> None:
        """Set the direct lattice vectors expressed as 4-index Weber
        symbols.
        """
        self.uvw = _UVTW2uvw(value)

    @property
    def u(self) -> np.ndarray:
        """Return the first direct lattice vector index."""
        return self.uvw[..., 0]

    @property
    def v(self) -> np.ndarray:
        """Return the second direct lattice vector index."""
        return self.uvw[..., 1]

    @property
    def w(self) -> np.ndarray:
        """Return the third direct lattice vector index."""
        return self.uvw[..., 2]

    @property
    def U(self) -> np.ndarray:
        r"""Return the first direct lattice vector index in 4-index
        Weber symbols, equal to :math:`(2u - v)/(3)`.
        """
        return self.UVTW[..., 0]

    @property
    def V(self) -> np.ndarray:
        r"""Return the second direct lattice vector index in 4-index
        Weber symbols, equal to :math:`(2v - u)/3`.
        """
        return self.UVTW[..., 1]

    @property
    def T(self) -> np.ndarray:
        r"""Return the third direct lattice vector index in 4-index
        Weber symbols, equal to :math:`-(u + v)/3`.
        """
        return self.UVTW[..., 2]

    @property
    def W(self) -> np.ndarray:
        r"""Return the fourth direct lattice vector index in 4-index
        Weber symbols, equal to :math:`w`.
        """
        return self.UVTW[..., 3]

    @property
    def length(self) -> np.ndarray:
        """Return the length of each vector given in lattice parameter
        units if the :attr:`coordinate_format` attribute equals
        ``"uvw"`` or ``"UVTW"``, and inverse lattice parameter units if
        the attribute equals ``"hkl"`` or ``"hkil"``.

        If the attribute equals ``"xyz"``, the norms of the vectors in
        :attr:`data` are returned.
        """
        if self.coordinate_format in ["hkl", "hkil"]:
            return self.phase.structure.lattice.rnorm(self.hkl)
        elif self.coordinate_format in ["uvw", "UVTW"]:
            return self.phase.structure.lattice.norm(self.uvw)
        else:
            return self.norm

    @property
    def multiplicity(self) -> np.ndarray:
        """Return the number of symmetrically equivalent directions per
        vector.
        """
        _, l = self.symmetrise(unique=True, return_multiplicity=True)
        return l.reshape(self.shape)

    @property
    def space(self) -> str:
        """Return whether the vector is in direct (``"d"``) or
        reciprocal (``"r"``) space.
        """
        if self.coordinate_format in ["xyz", "uvw", "UVTW"]:
            return "d"
        else:
            return "r"

    @property
    def is_hexagonal(self) -> bool:
        """Return whether the crystal reference frame is
        hexagonal/trigonal.
        """
        return self.phase.is_hexagonal

    @property
    def unit(self) -> Self:
        """Return unit vectors."""
        m = self.__class__(xyz=super().unit.data, phase=self.phase)
        m.coordinate_format = self.coordinate_format
        return m

    # ------------------------ Dunder methods ------------------------ #

    def __repr__(self) -> str:
        """String representation."""
        name = self.__class__.__name__
        shape = self.shape
        symmetry = None if self.phase is None else self.phase.point_group.name
        coordinate_format = self.coordinate_format
        data = np.array_str(self.coordinates, precision=4, suppress_small=True)
        return (
            f"{name} {shape}, point group {symmetry}, {coordinate_format}\n" f"{data}"
        )

    def __getitem__(self, key) -> Self:
        """NumPy fancy indexing of vectors."""
        m = self.__class__(xyz=self.data[key], phase=self.phase).deepcopy()
        m.coordinate_format = self.coordinate_format
        return m

    # ------------------------ Class methods ------------------------- #

    @classmethod
    def from_highest_indices(
        cls,
        phase: "Phase",
        uvw: Union[np.ndarray, list, tuple, None] = None,
        hkl: Union[np.ndarray, list, tuple, None] = None,
        include_zero_vector: bool = False,
    ) -> Self:
        """Create a set of unique direct or reciprocal lattice vectors
        from three highest indices and a phase (crystal lattice and
        symmetry).

        Parameters
        ----------
        phase
            A phase with a crystal lattice and symmetry.
        uvw
            Three highest direct lattice vector indices.
        hkl
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
    def from_min_dspacing(cls, phase: "Phase", min_dspacing: float = 0.05) -> Self:
        """Create a set of unique reciprocal lattice vectors with a
        a direct space interplanar spacing greater than a lower
        threshold.

        Parameters
        ----------
        phase
            A phase with a crystal lattice and symmetry.
        min_dspacing
            Smallest interplanar spacing to consider. Default is 0.05,
            in the unit used to define the lattice parameters in
            ``phase``.
        """
        highest_hkl = _get_highest_hkl(
            lattice=phase.structure.lattice, min_dspacing=min_dspacing
        )
        hkl = _get_indices_from_highest(highest_indices=highest_hkl)
        hkl = hkl.astype(float).round(0)
        return cls(hkl=hkl, phase=phase).unique()

    @classmethod
    def random(
        cls,
        phase: "Phase",
        shape: Union[int, tuple] = 1,
        coordinate_format: str = "xyz",
    ) -> Self:
        """Create random Miller indices.

        Parameters
        ----------
        phase
            A phase with a crystal lattice and symmetry.
        shape
            Shape of the indices.
        coordinate_format
            Coordinate format of indices, either ``"xyz"`` (default),
            ``"uvw"``, ``"UVTW"``, ``"hkl"``, or ``"hkil"``.

        Returns
        -------
        m
            Random Miller indices.

        Examples
        --------
        >>> from orix.crystal_map import Phase
        >>> from orix.vector import Miller
        >>> phase = Phase(point_group="m-3m")
        >>> _ = Miller.random(phase)
        >>> _ = Miller.random(phase, (3, 4))
        >>> _ = Miller.random(phase, (3, 4), "hkl")
        """
        v = Vector3d.random(shape)
        m = Miller(xyz=v.data, phase=phase)
        m.coordinate_format = coordinate_format
        return m

    # --------------------- Other public methods --------------------- #

    def deepcopy(self) -> Self:
        """Return a deepcopy of the instance."""
        return deepcopy(self)

    def round(self, max_index: int = 20) -> Self:
        """Round a set of index triplet (Miller) or quartet
        (Miller-Bravais/Weber) to the *closest* smallest integers.

        Adopted from MTEX' :code:`Miller.round` function.

        Parameters
        ----------
        max_index
            Maximum integer index to round to, by default 20.

        Returns
        -------
        mill
            Rounded set of index triplet(s) or quartet(s).
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
    ) -> Union[Self, Tuple[Self, np.ndarray], Tuple[Self, np.ndarray, np.ndarray]]:
        """Return vectors symmetrically equivalent to the vectors.

        Parameters
        ----------
        unique
            Whether to return only unique vectors. Default is ``False``.
        return_multiplicity
            Whether to return the multiplicity of each vector. Default
            is ``False``.
        return_index
            Whether to return the index into the vectors for the
            returned symmetrically equivalent vectors. Default is
            ``False``.

        Returns
        -------
        m
            Flattened symmetrically equivalent vectors.
        multiplicity
            Multiplicity of each vector. Returned if
            ``return_multiplicity=True``.
        idx
            Index into the vectors for the returned symmetrically
            equivalent vectors. Returned if ``return_index=True``.
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

    def angle_with(
        self,
        other: Self,
        use_symmetry: bool = False,
        degrees: bool = False,
    ) -> np.ndarray:
        """Return the angles between these vectors and the other vectors
        possibly using symmetrically equivalent vectors to find the
        smallest angle under symmetry.

        Vectors must have compatible shapes, and be in the same space
        (direct or reciprocal) and crystal reference frames.

        Parameters
        ----------
        other
            Other vectors.
        use_symmetry
            Whether to consider equivalent vectors to find the smallest
            angle under symmetry. Default is ``False``.
        degrees
            If ``True``, the given angles are returned in degrees.
            Default is ``False``.

        Returns
        -------
        angles
            Angles in radians (``degrees=False``)  or degrees
            (``degrees=True``).
        """
        self._compatible_with(other, raise_error=True)

        if use_symmetry:
            other2 = other.symmetrise(unique=True)
            cosines = self.dot_outer(other2) / (
                self.norm[..., np.newaxis] * other2.norm[np.newaxis, ...]
            )
            cosines = np.round(cosines, 12)
            angles = np.min(np.arccos(cosines), axis=-1)
        else:
            angles = super().angle_with(other)

        if degrees:
            angles = np.rad2deg(angles)

        return angles

    def cross(self, other: Self) -> Self:
        """Return the cross products of the vectors with the other
        vectors, which is considered the zone axes between the vectors.

        Parameters
        ----------
        other
            Other vectors, which must be in the same space (direct or
            reciprocal) and have the same crystal reference frame.

        Returns
        -------
        m
            Vectors in reciprocal (direct) space if direct (reciprocal)
            vectors are crossed.
        """
        self._compatible_with(other, raise_error=True)
        new_fmt = dict(hkl="uvw", uvw="hkl", hkil="UVTW", UVTW="hkil")
        m = self.__class__(xyz=super().cross(other).data, phase=self.phase)
        m.coordinate_format = new_fmt[self.coordinate_format]
        return m

    def dot(self, other: Self) -> np.ndarray:
        """Return the dot products of the vectors and the other vectors.

        Parameters
        ----------
        other
            Other vectors, which must be in the same space (direct or
            reciprocal) and have the same crystal reference frame.

        Returns
        -------
        dot_products
            Dot products.
        """
        self._compatible_with(other, raise_error=True)
        return super().dot(other)

    def dot_outer(self, other: Self) -> np.ndarray:
        """Return the outer dot products of the vectors and the other
        vectors.

        Parameters
        ----------
        other
            Other vectors, which must be in the same space (direct or
            reciprocal) and have the same crystal reference frame.

        Returns
        -------
        dot_products
            Dot products.
        """
        self._compatible_with(other, raise_error=True)
        return super().dot_outer(other)

    def flatten(self) -> Self:
        """Return the flattened vectors.

        Returns
        -------
        m
            Flattened vectors.
        """
        m = self.__class__(xyz=super().flatten().data, phase=self.phase)
        m.coordinate_format = self.coordinate_format
        return m

    def transpose(self, *axes: Optional[int]) -> Self:
        """Return a new instance with the data transposed.

        The order may be undefined if :attr:`ndim` is originally 2. In
        this case the first two dimensions are transposed.

        Parameters
        ----------
        axes
            Transposed axes order. Only navigation axes need to be
            defined. May be undefined if self only contains two
            navigation dimensions.

        Returns
        -------
        m
            New transposed Miller instance of the original instance.
        """
        m = self.__class__(xyz=super().transpose(*axes).data, phase=self.phase)
        m.coordinate_format = self.coordinate_format
        return m

    def get_nearest(self, *args) -> NotImplemented:
        """NotImplemented."""
        return NotImplemented

    def mean(self, use_symmetry: bool = False) -> Self:
        """Return the mean vector of the set of vectors.

        Parameters
        ----------
        use_symmetry
            Not implemented yet.

        Returns
        -------
        m
            Mean vector.
        """
        # TODO: Allow using symmetry by projecting to fundamental sector
        if use_symmetry:
            return NotImplemented
        m = self.__class__(xyz=super().mean().data, phase=self.phase)
        m.coordinate_format = self.coordinate_format
        return m

    def reshape(self, *shape: Union[int, tuple]) -> Self:
        """Return a new instance with the vectors reshaped.

        Parameters
        ----------
        *shape
            The new shape as one or more integers or as a tuple.

        Returns
        -------
        m
            New instance.
        """
        m = self.__class__(xyz=super().reshape(*shape).data, phase=self.phase)
        m.coordinate_format = self.coordinate_format
        return m

    def unique(
        self, use_symmetry: bool = False, return_index: bool = False
    ) -> Union[Self, Tuple[Self, np.ndarray]]:
        """Unique vectors in ``self``.

        Parameters
        ----------
        use_symmetry
            Whether to consider equivalent vectors to compute the unique
            vectors. Default is ``False``.
        return_index
            Whether to return the indices of the (flattened) data where
            the unique entries were found. Default is ``False``.

        Returns
        -------
        m
            Flattened unique vectors.
        idx
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
            v2 = operations.outer(v).flatten().reshape(n_v, operations.size)
            data = v2.data.round(10)
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

    def in_fundamental_sector(self, symmetry: Optional["Symmetry"] = None) -> Self:
        """Project Miller indices to a symmetry's fundamental sector
        (inverse pole figure).

        This projection is taken from MTEX'
        :code:`project2FundamentalRegion`.

        Parameters
        ----------
        symmetry
            Symmetry with a fundamental sector, possibly not equal to
            :attr:`~orix.crystal_map.Phase.point_group`. If not given,
            ``point_group`` is used if valid, otherwise an error is
            raised.

        Returns
        -------
        m
            Vectors within the fundamental sector.

        Examples
        --------
        >>> from orix.crystal_map import Phase
        >>> from orix.quaternion.symmetry import D6h
        >>> from orix.vector import Miller
        >>> t = Miller(uvw=(-1, 1, 0), phase=Phase(point_group="m-3m"))
        >>> t.in_fundamental_sector()
        Miller (1,), point group m-3m, uvw
        [[1. 0. 1.]]
        >>> t.in_fundamental_sector(D6h)
        Miller (1,), point group m-3m, uvw
        [[1.366 0.366 0.   ]]
        """
        if symmetry is None:
            symmetry = self.phase.point_group
            if symmetry is None:
                raise ValueError(
                    "`symmetry` must be passed or `self.phase.point_group` must be a "
                    "`Symmetry` with a `Symmetry.fundamental_sector`"
                )
        v = Vector3d(self.data).in_fundamental_sector(symmetry)
        m = self.__class__(xyz=v.data, phase=self.phase)
        m.coordinate_format = self.coordinate_format
        return m

    # -------------------- Other private methods --------------------- #

    def _compatible_with(self, other: Self, raise_error: bool = False) -> bool:
        """Whether ``self`` and ``other`` are the same (the same crystal
        lattice and symmetry) with vectors in the same space.

        Parameters
        ----------
        other
            Another vector instance.
        raise_error
            Whether to raise a ``ValueError`` if the instances are
            incompatible (default is ``False``).

        Returns
        -------
        compatible
            Whether they are compatible.
        """
        same_symmetry = self.phase.point_group == other.phase.point_group
        same_lattice = np.allclose(
            self.phase.structure.lattice.abcABG(),
            other.phase.structure.lattice.abcABG(),
        )
        same_space = self.space == other.space
        compatible = same_symmetry * same_lattice * same_space
        if not compatible and raise_error:
            raise ValueError(
                "The crystal lattices and symmetries must be the same, and the "
                "vector(s) must be in the same space"
            )
        else:
            return compatible


def _transform_space(
    v_in: np.ndarray, space_in: str, space_out: str, lattice: Lattice
) -> np.ndarray:
    r"""Convert vectors in a unit cell from one space to another.

    Parameters
    ----------
    v_in
        Input vectors.
    space_in, space_out
        ``"d"`` for direct (uvw), ``"r"`` for reciprocal (hkl) or
        ``"c"`` for cartesian (xyz).

    Returns
    -------
    v_out
        Output vectors.

    Notes
    -----
    Conversions between direct lattice vectors :math:`[uvw]`, reciprocal
    lattice vectors :math:`(hkl)` and Cartesian vectors
    :math:`(x, y, z)` using the structure matrix :math:`A` and the
    metric tensor :math:`g_{ij}`, where vectors are row vectors and
    matrices are row matrices:

    .. math::

        (x, y, z) = [uvw] \cdot \mathbf{A}
        (x, y, z) = (hkl) \cdot (\mathbf{A}^{-1})^T
        [uvw] = (x, y, z) \cdot \mathbf{A}^{-1}
        [uvw] = (hkl) \cdot g_{ij}^{-1}
        (hkl) = (x, y, z) \cdot A^T
        (hkl) = [uvw] \cdot g_{ij}
    """
    spaces = ["d", "r", "c"]
    if space_in not in spaces or space_out not in spaces:
        raise ValueError(f"`space_in` and `space_out` must be one of {spaces}")

    if space_in == space_out:
        v_out = np.copy(v_in)
    elif space_in == "d":
        if space_out == "c":
            # xyz = uvw * A
            v_out = np.matmul(v_in, lattice.base)
        else:
            # hkl = uvw * g_ij
            v_out = np.matmul(v_in, lattice.metrics)
    elif space_in == "r":
        if space_out == "c":
            # xyz = hkl * (A^-1)^T
            v_out = np.matmul(v_in, lattice.recbase.T)
        else:
            # uvw = hkl * g_ij^-1
            v_out = np.matmul(v_in, lattice.reciprocal().metrics)
    else:
        if space_out == "d":
            # uvw = xyz * A^-1
            v_out = np.dot(v_in, lattice.recbase)
        else:
            # hkl = xyz * ((A^-1)^T)^-1 = xyz * A^T
            v_out = np.matmul(v_in, lattice.base.T)

    return v_out


def _hkl2hkil(hkl: np.ndarray) -> np.ndarray:
    hkl = np.asarray(hkl)
    hkil = np.zeros(hkl.shape[:-1] + (4,))
    h = hkl[..., 0]
    k = hkl[..., 1]
    hkil[..., 0] = h
    hkil[..., 1] = k
    hkil[..., 2] = -(h + k)
    hkil[..., 3] = hkl[..., 2]
    return hkil


def _hkil2hkl(hkil: np.ndarray) -> np.ndarray:
    hkil = np.asarray(hkil)
    hkl = np.zeros(hkil.shape[:-1] + (3,))
    hkl[..., :2] = hkil[..., :2]
    hkl[..., 2] = hkil[..., 3]
    return hkl


def _check_hkil(hkil: np.ndarray) -> None:
    hkil = np.asarray(hkil)
    if not np.allclose(np.sum(hkil[..., :3], axis=-1), 0, atol=1e-4):
        raise ValueError(
            "The Miller-Bravais indices convention h + k + i = 0 is not satisfied"
        )


def _uvw2UVTW(uvw: np.ndarray, convention: Optional[str] = None) -> np.ndarray:
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


def _UVTW2uvw(UVTW: np.ndarray, convention: Optional[str] = None) -> np.ndarray:
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


def _check_UVTW(UVTW: np.ndarray) -> None:
    UVTW = np.asarray(UVTW)
    if not np.allclose(np.sum(UVTW[..., :3], axis=-1), 0, atol=1e-4):
        raise ValueError(
            "The Miller-Bravais indices convention U + V + T = 0 is not satisfied"
        )


def _get_indices_from_highest(
    highest_indices: Union[list, tuple, np.ndarray]
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


def _get_highest_hkl(lattice: Lattice, min_dspacing: float = 0.05) -> np.ndarray:
    """Return the highest Miller indices hkl of the plane with a direct
    space interplanar spacing closest to a threshold.

    Parameters
    ----------
    lattice
        Crystal lattice.
    min_dspacing
        Smallest interplanar spacing to consider. Default is 0.05.

    Returns
    -------
    highest_hkl
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


def _round_indices(
    indices: Union[list, tuple, np.ndarray], max_index: int = 12
) -> np.ndarray:
    """Round a set of index triplet (Miller) or quartet (Miller-Bravais)
    to the *closest* smallest integers.

    Adopted from MTEX' :code:`Miller.round` function.

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
        / np.sum(idx_scaled**2, axis=-1)
    )
    idx_min_error = np.argmin(error, axis=0)
    multiplier = (idx_min_error + 1) / max_per_set

    # Reshape `multiplier` to match indices shape
    multiplier = multiplier.reshape(idx.shape[:-1])[..., np.newaxis]

    # Finally, multiply each set with their most suitable multiplier,
    # and round
    new_indices = np.round(multiplier * idx).astype(int)

    return new_indices
