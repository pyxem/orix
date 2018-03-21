import numpy as np
import itertools


def check(obj, cls):
    if not isinstance(obj, cls):
        try:
            obj = cls(obj)
        except:
            raise ValueError('Could not turn {} (type {}) into {}'.format(
                        obj, obj.__class__.__name__, cls.__name__))
    return obj


class DimensionError(Exception):

    def __init__(self, object):
        class_name = object.__class__.__name__
        required_dimension = object.dim
        received_dimension = object.shape[-1]
        super().__init__(
            "{} requires data of dimension {} "
            "but received dimension {}.".format(
                class_name, required_dimension, received_dimension
            ))


class Object3d:
    """Base class for 3d objects.

    """

    dim = None
    data = None
    plot_type = None
    __array_ufunc__ = None

    def __init__(self, data=None):
        if isinstance(data, Object3d):
            data = data.data
        data = np.atleast_2d(data)
        self.data = data
        if data.shape[-1] != self.dim:
            raise DimensionError(self)

    def __repr__(self):
        name = self.__class__.__name__
        shape = str(self.shape)
        data = np.array_str(self.data, precision=4, suppress_small=True)
        return '\n'.join([name + ' ' + shape, data])

    def __getitem__(self, key):
        data = self.data[key]
        obj = self.__class__(data)
        return obj

    def __setitem__(self, key, value):
        self.data[key] = value.data

    @classmethod
    def empty(cls):
        """Creates an empty object with the appropriate dimensions."""
        return cls(np.zeros((0, cls.dim)))

    @property
    def shape(self):
        return self.data.shape[:-1]

    @property
    def data_dim(self):
        """int : The dimensions of `data`.

        For example, if `data` has shape (4, 4, 3), `data_dim` is 3.

        """
        return len(self.shape)

    @property
    def size(self):
        """The total number of entries in this object."""
        return np.prod(self.shape)

    @classmethod
    def stack(cls, sequence):
        sequence = [check(s, cls).data for s in sequence]
        try:
            stack = np.stack(sequence, axis=-2)
        except ValueError:
            raise
        return cls(stack)

    def flatten(self):
        """Returns a new object containing the same data in a single column."""
        return self.__class__(self.data.T.reshape(self.dim, -1).T)

    def unique(self, return_index=False, return_inverse=False):
        """Returns a new object containing only this object's unique entries.

        Unless overridden, this method returns the numerically unique entries.
        Also removes zero entries which are assumed to be degenerate.

        Parameters
        ----------
        return_index : bool, optional
            If True, will also return the indices of the (flattened) data where
            the unique entries were found.
        return_inverse : bool, optional
            If True, will also return the indices to reconstruct the (flattened)
            data from the unique data.

        Returns
        -------
        dat : Object3d
            The numerically unique entries.
        idx : numpy.ndarray, optional
            The indices of the unique data in the (flattened) array.
        inv : numpy.ndarray, optional
            The indices of the (flattened) data in the unique array.

        """
        data = self.flatten().data.round(9)
        # Remove zeros
        data = data[~np.all(np.isclose(data, 0), axis=1)]
        if len(data) == 0:
            return self.__class__(data)
        _, idx, inv = np.unique(data.round(4), axis=0, return_index=True, return_inverse=True)
        dat = self.__class__(data[np.sort(idx)])
        if return_index and return_inverse:
            return dat, idx, inv
        elif return_index and not return_inverse:
            return dat, idx
        elif return_inverse and not return_index:
            return dat, inv
        else:
            return dat

    @property
    def norm(self):
        from texpy.scalar import Scalar
        return Scalar(np.sqrt(np.sum(np.square(self.data), axis=-1)))

    @property
    def unit(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.__class__(np.nan_to_num(self.data / self.norm.data[..., np.newaxis]))

    def numerical_sort(self):
        dat = self.data.round(4)
        ind = np.lexsort([dat[:, i] for i in range(self.dim - 1, -1, -1)])
        return self.__class__(self.data[ind])

    def reshape(self, *args):
        """Returns a new object containing the same data with a new shape."""
        return self.__class__(self.data.reshape(*args, self.dim))

    def plot(self, ax=None, figsize=(6, 6), **kwargs):
        plt = self.plot_type(self, ax=ax, figsize=figsize)
        plt.draw(**kwargs)
        return plt.ax

    def sample(self, n=500):
        """Selects 'n' random values of this data.

        Parameters
        ----------
        n : int
            The number of samples to draw from this object.

        """
        if n > self.size:
            n = self.size
        sample = np.random.choice(self.size, n, False)
        return self.flatten()[sample]
