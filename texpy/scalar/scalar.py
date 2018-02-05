import numpy as np

from texpy.base.object3d import Object3d
from texpy.plot.scalar_plot import ScalarPlot


class Scalar(Object3d):
    """Zero-dimensional data objects.

    Scalars represent single numerical values, such as the x-component of a
    vector or the rotation angle of a quaternion.

    Scalars support mathematical operations with sane data types including
    numpy arrays, integers, vectors, etc.

    Attributes
    ----------
    data : ndarray

    """

    dim = 0
    data = None
    plot_type = ScalarPlot

    def __init__(self, data):
        if isinstance(data, self.__class__):
            data = data.data
        data = np.atleast_1d(data)
        self.data = data

    def __neg__(self):
        return self.__class__(-self.data)

    def __add__(self, other):
        if isinstance(other, Scalar):
            return self.__class__(self.data + other.data)
        elif isinstance(other, (int, float)):
            return self.__class__(self.data + other)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(self.data + other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Scalar):
            return self.__class__(other.data + self.data)
        elif isinstance(other, (int, float)):
            return self.__class__(other + self.data)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(other + self.data)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Scalar):
            return self.__class__(self.data - other.data)
        elif isinstance(other, (int, float)):
            return self.__class__(self.data - other)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(self.data - other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Scalar):
            return self.__class__(other.data - self.data)
        elif isinstance(other, (int, float)):
            return self.__class__(other - self.data)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(other - self.data)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Scalar):
            return self.__class__(self.data * other.data)
        elif isinstance(other, (int, float)):
            return self.__class__(self.data * other)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(self.data * other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Scalar):
            return self.__class__(other.data * self.data)
        elif isinstance(other, (int, float)):
            return self.__class__(other * self.data)
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.__class__(other * self.data)
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Object3d):
            return self.data > other.data
        elif isinstance(other, (int, float)):
            return self.data > other
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.data > other
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Object3d):
            return self.data < other.data
        elif isinstance(other, (int, float)):
            return self.data < other
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.data < other
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Object3d):
            return self.data >= other.data
        elif isinstance(other, (int, float)):
            return self.data >= other
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.data >= other
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Object3d):
            return self.data <= other.data
        elif isinstance(other, (int, float)):
            return self.data <= other
        elif isinstance(other, (list, tuple)):
            other = np.array(other)
        if isinstance(other, np.ndarray):
            return self.data <= other
        return NotImplemented

    def __pow__(self, power, modulo=None):
        if isinstance(power, (int, float)):
            return self.__class__(self.data ** power)
        elif isinstance(power, (list, tuple)):
            power = np.array(power)
        if isinstance(power, np.ndarray):
            return self.__class__(self.data ** power)
        return NotImplemented

    @classmethod
    def stack(cls, sequence):
        sequence = [s.data for s in sequence]
        try:
            stack = np.stack(sequence, axis=-1)
        except ValueError:
            raise
        return cls(stack)

    @property
    def shape(self):
        return self.data.shape