import numpy as np
import itertools


def check_matching_type(this, obj):
    if not isinstance(obj, this.__class__):
        try:
            obj = this.__class__(obj)
        except:
            raise ValueError('Could not turn {} (type {}) into {}'.format(
                        obj, obj.__class__.__name__, this.__class__.__name__))
    return obj


class Object3d:

    dim = None
    data = None
    __array_priority__ = 1000

    def __init__(self, data=None):
        if isinstance(data, self.__class__):
            data = data.data
        data = np.atleast_2d(data)
        assert data.shape[-1] == self.dim, "Data must have final shape (..., {}). Got shape {}.".format(self.dim, data.shape)
        self.data = data

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
        return cls(np.zeros((0, cls.dim)))

    @property
    def shape(self):
        return self.data.shape[:-1]

    @property
    def data_dim(self):
        return len(self.shape)

    @property
    def size(self):
        s = 1
        for i in self.shape:
            s *= i
        return s

    def outer(self, other):
        data = np.zeros(self.shape + other.shape + (other.dim,))
        for i, j in itertools.product(np.ndindex(self.shape), np.ndindex(other.shape)):
            data[i + j] = (self[i] * other[j]).data
        # data = np.squeeze(data)
        return other.__class__(data)

    @classmethod
    def stack(cls, sequence):
        s0 = sequence[0]
        sequence = [check_matching_type(s0, s).data for s in sequence]
        try:
            stack = np.stack(sequence, axis=-2)
        except ValueError:
            raise
        return cls(stack)

    def flatten(self):
        return self.__class__(self.data.T.reshape(self.dim, -1).T)

    def unique(self, return_index=False, return_inverse=False):
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
        return np.sqrt(np.sum(np.square(self.data), axis=-1))

    @property
    def unit(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.__class__(np.nan_to_num(self.data / self.norm[..., np.newaxis]))

    def numerical_sort(self):
        dat = self.data.round(4)
        ind = np.lexsort([dat[:, i] for i in range(self.dim - 1, -1, -1)])
        return self.__class__(self.data[ind])

    def reshape(self, *args):
        return self.__class__(self.data.reshape(*args, self.dim))

