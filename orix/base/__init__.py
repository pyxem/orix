# -*- coding: utf-8 -*-
# Copyright 2018-2022 the orix developers
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

"""Base class for 3d objects.

Note that this class is not meant to be used directly.
"""

import numpy as np

from orix._util import deprecated


# Lists what will be imported when calling "from orix.base import *"
__all__ = [
    "check",
    "DimensionError",
    "Object3d",
]


def check(obj, cls):
    if not isinstance(obj, cls):
        try:
            obj = cls(obj)
        except BaseException:
            raise ValueError(
                "Could not turn {} (type {}) into {}".format(
                    obj, obj.__class__.__name__, cls.__name__
                )
            )
    return obj


class DimensionError(Exception):
    def __init__(self, this, data):
        class_name = this.__class__.__name__
        required_dimension = this.dim
        received_dimension = data.shape[-1]
        super().__init__(
            "{} requires data of dimension {} "
            "but received dimension {}.".format(
                class_name, required_dimension, received_dimension
            )
        )


class Object3d:
    """Base class for 3d objects."""

    dim = None
    """int : The number of dimensions for this object."""  # pragma: no cover

    _data = None
    """numpy.ndarray : Array holding this object's numerical data."""  # pragma: no cover

    __array_ufunc__ = None

    def __init__(self, data=None):
        if isinstance(data, Object3d):
            self._data = data.data
        else:
            data = np.atleast_2d(data)
            if data.shape[-1] != self.dim:
                raise DimensionError(self, data)
            self._data = data
        self.__finalize__(data)

    def __finalize__(self, data):
        pass

    @property
    def data(self):
        return self._data[..., : self.dim]

    @data.setter
    def data(self, data):
        self._data[..., : self.dim] = data

    def __repr__(self):
        name = self.__class__.__name__
        shape = str(self.shape)
        data = np.array_str(self.data, precision=4, suppress_small=True)
        return "\n".join([name + " " + shape, data])

    def __getitem__(self, key):
        data = np.atleast_2d(self.data[key])
        obj = self.__class__(data)
        return obj

    def __setitem__(self, key, value):
        self.data[key] = value.data

    @classmethod
    def empty(cls):
        """An empty object with the appropriate dimensions."""
        return cls(np.zeros((0, cls.dim)))

    @property
    def shape(self):
        """tuple : Shape of the object."""
        return self.data.shape[:-1]

    @property
    @deprecated(since="0.8", alternative="ndim", removal="0.9", object_type="property")
    def data_dim(self):
        """int : The dimensions of the data.

        For example, if `data` has shape (4, 5, 6), `data_dim` is 3.

        """
        return self.ndim

    @property
    def ndim(self):
        """int : The number of navigation dimensions of the instance.

        For example, if `data` has shape (4, 5, 6), `data_dim` is 3.

        """
        return len(self.shape)

    @property
    def size(self):
        """int : Total number of entries in this object."""
        return np.prod(self.shape)

    @classmethod
    def stack(cls, sequence):
        sequence = [s._data for s in sequence]
        stack = np.stack(sequence, axis=-2)
        obj = cls(stack[..., : cls.dim])
        obj._data = stack
        return obj

    def flatten(self):
        """Returns a new object with the same data in a single column."""
        obj = self.__class__(self.data.T.reshape(self.dim, -1).T)
        real_dim = self._data.shape[-1]
        obj._data = self._data.T.reshape(real_dim, -1).T
        return obj

    def unique(self, return_index=False, return_inverse=False):
        """Returns a new object containing only this object's unique
        entries.

        Unless overridden, this method returns the numerically unique
        entries. Also removes zero entries which are assumed to be
        degenerate.

        Parameters
        ----------
        return_index : bool, optional
            If True, will also return the indices of the (flattened)
            data where the unique entries were found.
        return_inverse : bool, optional
            If True, will also return the indices to reconstruct the
            (flattened) data from the unique data.

        Returns
        -------
        dat : Object3d
            The numerically unique entries.
        idx : numpy.ndarray, optional
            The indices of the unique data in the (flattened) array.
        inv : numpy.ndarray, optional
            The indices of the (flattened) data in the unique array.

        """
        data = self.flatten()._data.round(12)
        data = data[~np.all(np.isclose(data, 0), axis=1)]  # Remove zeros
        _, idx, inv = np.unique(data, axis=0, return_index=True, return_inverse=True)
        obj = self.__class__(data[np.sort(idx), : self.dim])
        obj._data = data[np.sort(idx)]
        if return_index and return_inverse:
            return obj, idx, inv
        elif return_index and not return_inverse:
            return obj, idx
        elif return_inverse and not return_index:
            return obj, inv
        else:
            return obj

    @property
    def norm(self):
        return np.sqrt(np.sum(np.square(self.data), axis=-1))

    @property
    def unit(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            obj = self.__class__(np.nan_to_num(self.data / self.norm[..., np.newaxis]))
            return obj

    def squeeze(self):
        """Returns a new object with length one dimensions removed."""
        obj = self.__class__(self)
        obj._data = np.atleast_2d(np.squeeze(self._data))
        return obj

    def reshape(self, *shape):
        """Returns a new object containing the same data with a new shape."""
        obj = self.__class__(self.data.reshape(*shape, self.dim))
        obj._data = self._data.reshape(*shape, -1)
        return obj

    def transpose(self, *axes):
        """Returns a new object containing the same data transposed.

        If ndim is originally 2, then order may be undefined.
        In this case the first two dimensions will be transposed.

        Parameters
        ----------
        axes : int, optional
            The transposed axes order. Only navigation axes need to be
            defined. May be undefined if self only contains two
            navigation dimensions.

        Returns
        -------
        obj :
            A transposed instance of the object.

        """
        # 1d object should not be transposed
        if len(self.shape) == 1:
            return self

        # allow 2d object to be transposed without specifying axes
        if not len(axes):
            if len(self.shape) != 2:
                raise ValueError("Axes must be defined for more than two dimensions.")
            else:
                # swap first two axes
                axes = (1, 0)

        if len(axes) != len(self.shape):
            raise ValueError(
                "Number of axes is ill-defined: "
                + f"{tuple(axes)} does not fit with {self.shape}."
            )

        return self.__class__(self.data.transpose(*axes, -1))

    def get_plot_data(self):
        return self

    def get_random_sample(self, size=1, replace=False, shuffle=False):
        """Return a random sample of a given size in a flattened
        instance.

        Parameters
        ----------
        size : int, optional
            Number of samples to draw. Cannot be greater than the size
            of this instance. If not given, a single sample is drawn.
        replace : bool, optional
            See :meth:`numpy.random.Generator.choice`.
        shuffle : bool, optional
            See :meth:`numpy.random.Generator.choice`.

        Returns
        -------
        new
            New, flattened instance of a given size with elements drawn
            randomly from this instance.

        See Also
        --------
        numpy.random.Generator.choice
        """
        n = self.size
        if size > n:
            raise ValueError(f"Cannot draw a sample greater than {self.size}")
        rng = np.random.default_rng()
        sample = rng.choice(n, size=size, replace=replace, shuffle=shuffle)
        sample = np.unravel_index(sample, shape=self.shape)
        return self[sample]
