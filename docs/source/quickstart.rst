Getting Started
---------------

texpy is a package for handling three-dimensional objects such as vectors
and quaternions, in particular where these quantities arise from
spatially-resolved datasets, such as the orientation of crystals in a
scanned sample, or the vector of fluid motion in a box simulation. As implied
by the name, the main expected area of interest is for crystallographic
texture analysis.

Whether quaternions, vectors, or scalars, texpy quantities are represented
as arrays of objects, and a few methods are common to all.

Construction
~~~~~~~~~~~~

All texpy objects can be initialized from raw data in the form of lists,
tuples, or numpy arrays. In addition, compatible data types can be formed
from one another, so that :obj:`~texpy.quaternion.Quaternion` objects can be
converted to :obj:`~texpy.quaternion.rotation.Rotation` objects, for example.
The following code demonstrates a number of ways objects can be initialized.

.. ipython::
   :okexcept:

   In [1]: import numpy as np

   In [2]: from texpy.vector import Vector3d

   In [3]: v = Vector3d((1, 1, -2))

   In [4]: v  # A vector created from a tuple
   Out[4]:
   Vector3d (1,)
   [[ 1  1 -2]]

   In [5]: v = Vector3d([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

   In [6]: v  # A vector created from a list of lists
   Out[6]:
   Vector3d (3,)
   [[0 0 1]
    [0 1 0]
    [1 0 0]]

   In [7]: w = Vector3d(v)

   In [8]: w  # A vector created from another vector
   Out[8]:
   Vector3d (3,)
   [[0 0 1]
    [0 1 0]
    [1 0 0]]

   In [9]: v is w  # The new vector is a copy, not a view on the same object
   Out[9]: False

   In [10]: w = Vector3d(np.random.random((4, 4)))  # Incorrect dimension input
   ---------------------------------------------------------------------------
   DimensionError                            Traceback (most recent call last)
   <ipython-input-10-422ccfdd4ab4> in <module>()
   ----> 1 w = Vector3d(np.random.random((4, 4)))

   ~\Documents\phd\dev\texpy\texpy\base\__init__.py in __init__(self, data)
        41         self.data = data
        42         if data.shape[-1] != self.dim:
   ---> 43             raise DimensionError(self)
        44
        45     def __repr__(self):

   DimensionError: Vector3d requires data of dimension 3 but received dimension 4.

   In [11]: w = Vector3d(np.random.random((4, 3)))

   In [12]: w
   Out[12]:
   Vector3d (4,)
   [[0.1766 0.8614 0.6989]
    [0.3578 0.2829 0.6399]
    [0.5863 0.3862 0.0422]
    [0.5205 0.6528 0.4024]]

   In [13]: from texpy.vector.neo_euler import Rodrigues

   In [14]: Rodrigues(w)  # Converting between vector types
   Out[14]:
   Rodrigues (4,)
   [[0.1766 0.8614 0.6989]
    [0.3578 0.2829 0.6399]
    [0.5863 0.3862 0.0422]
    [0.5205 0.6528 0.4024]]

In addition, many texpy objects can be created from other types of data,
or from other texpy objects with a different parametrisation. For example, a
:obj:`~texpy.quaternion.rotation.Rotation` object has methods :meth:`~texpy
.quaternion.rotation.Rotation.from_euler` to create from an array of Euler
angles and :meth:`~texpy.quaternion.rotation.Rotation.from_neo_euler` to
create from neo-Euler objects such as
:obj:`~texpy.vector.neo_euler.AxAngle`.


Array Manipulation
~~~~~~~~~~~~~~~~~~

All texpy arrays support slicing and indexing in the style of numpy.

.. ipython::

   In [1]: import numpy as np

   In [2]: from texpy.quaternion import Quaternion

   In [3]: p = Quaternion(np.arange(3 * 4 * 4).reshape(3, 4, 4))

   In [4]: p  # The complete object.
   Out[4]:
   Quaternion (3, 4)
   [[[0.843  0.5158 0.0848 0.4627]
     [0.2016 0.0995 0.055  0.437 ]
     [0.8174 0.2794 0.3594 0.1949]
     [0.9363 0.1687 0.9187 0.1107]]

    [[0.1842 0.7484 0.6205 0.7538]
     [0.152  0.2224 0.4209 0.6535]
     [0.6419 0.0758 0.8169 0.7772]
     [0.4576 0.6627 0.7778 0.3165]]

    [[0.7514 0.4449 0.328  0.5949]
     [0.1778 0.8061 0.514  0.3119]
     [0.7899 0.8357 0.3773 0.5401]
     [0.1832 0.2562 0.867  0.021 ]]]

   In [5]: p[0]  # The first "row".
   Out[5]:
   Quaternion (4,)
   [[0.843  0.5158 0.0848 0.4627]
    [0.2016 0.0995 0.055  0.437 ]
    [0.8174 0.2794 0.3594 0.1949]
    [0.9363 0.1687 0.9187 0.1107]]

   In [6]: p[1]  # The second "row".
   Out[6]:
   Quaternion (4,)
   [[0.1842 0.7484 0.6205 0.7538]
    [0.152  0.2224 0.4209 0.6535]
    [0.6419 0.0758 0.8169 0.7772]
    [0.4576 0.6627 0.7778 0.3165]]

   In [7]: p[:, 2]  # All "rows", and the third "column".
   Out[7]:
   Quaternion (3,)
   [[0.8174 0.2794 0.3594 0.1949]
    [0.6419 0.0758 0.8169 0.7772]
    [0.7899 0.8357 0.3773 0.5401]]

   In [8]: p[0, 1]  # The first "row" and the second "column".
   Out[8]:
   Quaternion (1,)
   [[0.2016 0.0995 0.055  0.437 ]]

   In [9]: p[1:, 2]  # "Rows" 2 onwards, and the second "column" only.
   Out[9]:
   Quaternion (2,)
   [[0.6419 0.0758 0.8169 0.7772]
    [0.7899 0.8357 0.3773 0.5401]]

And so on.

.. important::

   MATLAB users should be aware that in Python, index counts start at 0, not 1!

A useful trick is indexing using boolean arrays. Entries will be returned
where the boolean index array is ``True`` only. Note that this array must be
of compatible shape with the array being indexed.

.. ipython::

   In [10]: mask = np.array([True, False, True])

   In [11]: p[mask]
   Out[11]:
   Quaternion (2, 4)
   [[[0.843  0.5158 0.0848 0.4627]
     [0.2016 0.0995 0.055  0.437 ]
     [0.8174 0.2794 0.3594 0.1949]
     [0.9363 0.1687 0.9187 0.1107]]

    [[0.7514 0.4449 0.328  0.5949]
     [0.1778 0.8061 0.514  0.3119]
     [0.7899 0.8357 0.3773 0.5401]
     [0.1832 0.2562 0.867  0.021 ]]]

   In [12]: mask = np.array([False, True, True, False])

   In [13]: p[:, mask]
   Out[13]:
   Quaternion (3, 2)
   [[[0.2016 0.0995 0.055  0.437 ]
     [0.8174 0.2794 0.3594 0.1949]]

    [[0.152  0.2224 0.4209 0.6535]
     [0.6419 0.0758 0.8169 0.7772]]

    [[0.1778 0.8061 0.514  0.3119]
     [0.7899 0.8357 0.3773 0.5401]]]

   In [14]: mask = np.array([[True, False, False, False], [False, False, True, True], [False, False, False, False]])

   In [15]: p[mask]
   Out[15]:
   Quaternion (3,)
   [[0.843  0.5158 0.0848 0.4627]
    [0.6419 0.0758 0.8169 0.7772]
    [0.4576 0.6627 0.7778 0.3165]]

.. note::

   Unlike numpy, indexing an array in texpy does not return a view on the
   original array. It creates a new object.


Plotting
~~~~~~~~

Most texpy objects can be plotted using the :meth:`texpy.base.Object3d.plot`
method. The type of plot depends on the object being plotted. For a more
complete explanation refer to :doc:`plotting`.


Maths
~~~~~

Most texpy objects are mathematical.

- :obj:`~texpy.scalar.Scalar` objects support operations like addition,
   subtraction, and so on.
- :obj:`~texpy.vector.Vector3d` objects can combine with scalars and other
   number-like objects in intuitive ways and also with each other - dot
   products and cross products are allowed.
- :obj:`texpy.quaternion.Quaternion` objects can be multiplied to
   vectors, but more importantly so can :obj:`texpy.quaternion.rotation.Rotation`
   objects, allowing vectors to be rotated. Quaternion objects can be
   multiplied together as well, and quaternion properties such as inversion and
   conjugation are accounted for.

For a complete description of each object's mathematical properties, refer
to the full :doc:`api`.

