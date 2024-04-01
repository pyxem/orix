.. _api:

=============
API reference
=============

**Release**: |version|

**Date**: |today|

This reference manual describes the public functions, modules, and objects in orix. Many
of the descriptions include brief examples. For learning how to use orix, see the
:doc:`/examples/index` or :doc:`/tutorials/index`.

.. caution::

    orix is in continuous development, meaning that some breaking changes and changes to
    this reference are likely with each release.

orix is organized in modules. This is the recommended way to import functionality from
the below list of modules:

.. autolink-skip::
.. code-block:: python

    >>> from orix.quaternion import Orientation, symmetry
    >>> O = Orientation.from_axes_angles([1, 1, 1], np.pi / 2, symmetry.Oh)
    >>> O
    Orientation (1,) m-3m
    [[0.7071 0.4082 0.4082 0.4082]]

Note that :mod:`numpy` and :mod:`matplotlib.pyplot` are available as ``np`` and ``plt``
in docstring examples, although they are not imported in order to reduce the number of
code lines.

.. currentmodule:: orix

.. rubric:: Modules

.. autosummary::
    :toctree: generated
    :template: custom-module-template.rst

    data
    crystal_map
    io
    measure
    plot
    projections
    quaternion
    sampling
    vector
