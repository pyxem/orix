# -*- coding: utf-8 -*-

import numpy as np

from typing import Dict, Optional, Tuple, Union

from orix.crystal_map.phase_list import Phase, PhaseList
from orix.quaternion import Orientation, Rotation
from orix.crystal_map import CrystalMap
from orix.vector import Miller, Vector3d

class CrystalMap3D(CrystalMap):
    """Crystallographic map of orientations, crystal phases and key
    properties associated with every spatial coordinate in 3D.

    Parameters
    ----------
    rotations
        Rotation in each point. Must be passed with all spatial
        dimensions in the first array axis (flattened). May contain
        multiple rotations per point, included in the second array
        axes. Crystal map data size is set equal to the first array
        axis' size.
    phase_id
        Phase ID of each pixel. IDs equal to ``-1`` are considered not
        indexed. If not given, all points are considered to belong to
        one phase with ID ``0``.
    x
        Map x coordinate of each data point. If not given, the map is
        assumed to be 1D, and it is set to an array of increasing
        integers from 0 to the length of the ``phase_id`` array.
    y
        Map y coordinate of each data point. If not given, the map is
        assumed to be 1D, and it is set to ``None``.
    z
        Map z coordinate of each data point. If not given, the map is
        assumed to be 1D, and it is set to ``None``.
    phase_list
        A list of phases in the data with their with names, space
        groups, point groups, and structures. The order in which the
        phases appear in the list is important, as it is this, and not
        the phases' IDs, that is used to link the phases to the
        input ``phase_id`` if the IDs aren't exactly the same as in
        ``phase_id``. If not given, a phase list with as many phases as
        there are unique phase IDs in ``phase_id`` is created.
    prop
        Dictionary of properties of each data point.
    scan_unit
        Length unit of the data. Default is ``"px"``.
    is_in_data
        Array of booleans signifying whether a point is in the data.
    """

    def __init__(
        self,
        rotations: "Rotation | CrystalMap",
        phase_id: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        z: Optional[np.ndarray] = None,
        phase_list: Optional[PhaseList] = None,
        prop: Optional[dict] = None,
        scan_unit: Optional[str] = "px",
        is_in_data: Optional[np.ndarray] = None,
    ):
        self._z = z

        super().__init__(rotations, phase_id, x, y, phase_list, prop, scan_unit,
                         is_in_data)
    
    # -------------------------- Properties -------------------------- #
    
    @property
    def z(self) -> Union[None, np.ndarray]:
        """Return the z coordinates of points in data."""
        if self._z is None or len(np.unique(self._z)) == 1:
            return
        else:
            return self._z[self.is_in_data]
        
    @property
    def dz(self) -> float:
        """Return the z coordinate step size."""
        return _step_size_from_coordinates(self._z)
    
    @property
    def row(self) -> Union[None, np.ndarray]:
        """Return the row coordinate of each point in the data.

        Returns ``None`` if :attr:`z` is not ``None``.

        Examples
        --------
        >>> from orix.crystal_map import CrystalMap
        >>> xmap = CrystalMap3D.empty((2, 3, 4))
        >>> xmap.row
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1])
        >>> xmap[1:3, 1:3, 1:3].row
        array([1, 1, 1, 1])
        """
        orig_shape = self._original_shape
        if len(orig_shape) == 1:
            if self.x is None:
                orig_shape += (1,)
            else:
                orig_shape = (1,) + orig_shape
        rows, _, _ = np.indices(orig_shape)
        rows = rows.flatten()[self.is_in_data]
        rows -= rows.min()
        return rows

    @property
    def col(self) -> Union[None, np.ndarray]:
        """Return the column coordinate of each point in the data.

        Returns ``None`` if :attr:`z` is not ``None``.

        Examples
        --------
        >>> from orix.crystal_map import CrystalMap
        >>> xxmap = CrystalMap3D.empty((2, 3, 4))
        >>> xmap.col
        array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2,
               2, 2, 2])
        >>> xmap[1:3, 1:3, 1:3].col
        array([1, 1, 2, 2])
        """
        shape = self._original_shape
        if len(shape) == 1:
            if self.x is None:
                shape += (1,)
            else:
                shape = (1,) + shape
        _, cols, _ = np.indices(shape)
        cols = cols.flatten()[self.is_in_data]
        cols -= cols.min()
        return cols
    
    @property
    def layer(self) -> Union[None, np.ndarray]:
        """Return the layer coordinate of each point in the data.

        Returns ``None`` if :attr:`z` is not ``None``.

        Examples
        --------
        >>> from orix.crystal_map import CrystalMap
        >>> xmap = CrystalMap3D.empty((2, 3, 4))
        >>> xmap.layer
        array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0,
               1, 2, 3])
        >>> xmap[1:3, 1:3, 1:3].layer
        array([1, 2, 1, 2])
        """
        orig_shape = self._original_shape
        if len(orig_shape) == 1:
            if self.x is None:
                orig_shape += (1,)
            else:
                orig_shape = (1,) + orig_shape
        _, _, layers = np.indices(orig_shape)
        layers = layers.flatten()[self.is_in_data]
        layers -= layers.min()
        return layers
    
    @property
    def _coordinates(self) -> dict:
        """Return the coordinates of points in the data."""
        return {"z": self.z, "y": self.y, "x": self.x}

    @property
    def _all_coordinates(self) -> dict:
        """Return the coordinates of all points."""
        return {"z": self._z, "y": self._y, "x": self._x}

    @property
    def _step_sizes(self) -> dict:
        """Return the step sizes of dimensions in the data."""
        return {"z": self.dz, "y": self.dy, "x": self.dx}
    
    
    # ------------------------ Class methods ------------------------- #


    @classmethod
    def slice2d(self, point, vector3d: Vector3d) -> CrystalMap:
        
        print('dummy function')
        
    @classmethod
    def slice2d_ortho(self, slice_index, axes_ortho) -> CrystalMap:
        """Return 2D CrystalMap sliced along grid/orthogonal plane to 3D"""
        
        print('dummy function')
        
        # if axes_ortho == 'xy':
        #     # [:, :, slice_index]
        #     print('asdf')

        # elif axes_ortho == 'xz':
        #     # [:, slice_index, :]
        #     print('asdf')

        # elif axes_ortho == 'yx':
        #     # [slice_index, :, :]
        #     print('asdf')

        # else:
        #     print('asdf')
       
        
       
    # ---------------------------------------------------------------- #

        
def _data_slices_from_coordinates(
    coords: Dict[str, np.ndarray], steps: Union[Dict[str, float], None] = None
) -> Tuple[slice]:
    """Return a list of slices defining the current data extent in all
    directions.

    Parameters
    ----------
    coords
        Dictionary with coordinate arrays.
    steps
        Dictionary with step sizes in each direction. If not given, they
        are computed from *coords*.

    Returns
    -------
    slices
        Data slice in each direction.
    """
    if steps is None:
        steps = {
            "x": _step_size_from_coordinates(coords["x"]),
            "y": _step_size_from_coordinates(coords["y"]),
            "z": _step_size_from_coordinates(coords["z"]),
        }
    slices = []
    for coords, step in zip(coords.values(), steps.values()):
        if coords is not None and step != 0:
            c_min, c_max = np.min(coords), np.max(coords)
            i_min = int(np.around(c_min / step))
            i_max = int(np.around((c_max / step) + 1))
            slices.append(slice(i_min, i_max))
    slices = tuple(slices)
    return slices

def _step_size_from_coordinates(coordinates: np.ndarray) -> float:
    """Return step size in input *coordinates* array.

    Parameters
    ----------
    coordinates
        Linear coordinate array.

    Returns
    -------
    step_size
        Step size in *coordinates* array.
    """
    unique_sorted = np.sort(np.unique(coordinates))
    if unique_sorted.size != 1:
        step_size = unique_sorted[1] - unique_sorted[0]
    else:
        step_size = 0
    return step_size