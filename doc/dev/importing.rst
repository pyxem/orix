Importing
=========

To keep the time for importing modules (such as :mod:`~orix.quaternion`) low, we use
`lazy_loader <https://scientific-python.org/specs/spec-0001/#lazy_loader>`__.
In practice, this means that when a user imports a module, class, or function, only the
functionality necessary is imported and cached, while "untouched" functionality is not
imported.

For example, when not lazy loading, all functions to be compiled with
:doc:`Numba <numba:index>` are compiled upon loading of most orix modules.
For example, quaternion module classes (:class:`~orix.quaternion.Orientation`,
:class:`~orix.quaternion.Rotation`, etc.) use functions leveraging Numba to quickly
convert to and from different conventions, such as
:meth:`~orix.quaternion.Orientation.from_euler` and
:meth:`~orix.quaternion.Orientation.to_rodrigues`.
These are fast, but require compilation before first use, which can take some time.
However, a user's script most likely does not need all these functions.
It makes sense, therefore, to only compile those needed, as they are needed.
Lazy loading takes care of this for the user, thus speeding up imports.

This time saving is also passed on to downsteam packages that import orix.
In most cases, this results in faster import times without additional work for
developers of these packages.

New imports go in the ``__init__.pyi`` "stub files", *not* in the ``__init__.py`` files.
By using stub files, static type checkers (e.g. in an advanced editor) are able to read
these files and infer information about the modules and functions (specifically, their
parameter and return "types"), without actually loading the modules themselves.

No imports should go in the ``__init__.py`` files except the lazy loading::

    import lazy_loader

    # Imports from stub file (see contributor guide for details)
    __getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

    del lazy_loader

The returns from lazy-loader are:

- ``__getattr__``: function to access names defined by the module
- ``__dir__``: list of names a module defines
- ``__all__``: list of module, class, and function names that should be imported when
  ``from package import *`` is encountered

We delete the lazy loader import at the end to prevent the possibility of importing it
from orix' modules.

Resources:

- `Lazy loading rationale <https://scientific-python.org/specs/spec-0001>`__
- `lazy-loader on GitHub <https://github.com/scientific-python/lazy-loader>`__
