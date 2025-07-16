Importing
=========

To keep the time for importing modules, such as :mod:`quaternion`, we use
`lazy_loader <https://scientific-python.org/specs/spec-0001/#lazy_loader>`__ (`GitHub
README <https://github.com/scientific-python/lazy-loader>`__).
In practice, this means that when a user imports a module, class, or function, only the
functionality necessary is imported and cached, while "untouched" functionality is not
imported.
For example, Numba functions not used in the functionality imported by the user are not
compiled, and libraries not used are not imported.

Another notable benefit to lazy loading is reduced import times for downstream packages.
In most cases, they only use parts of our functionality, and will thus not have to
import other parts they are not using.

New imports go in the ``__init__.pyi`` "stub files", *not* in the ``__init__.py`` files.
By using stub files, static type checkers are able to infer the types of lazy-loaded
modules and functions.
No imports should go in the ``__init__.py`` files except the lazy loading
functionality::

     import lazy_loader

     __getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

     del lazy_loader

The returns from lazy loader are:
- ``__getattr__``: function to access names defined by the module
- ``__dir__``: list of names a module defines
- ``__all__``: list of module, class, or function names that should be imported when
  ``from package import *`` is encountered

We delete the lazy loader import at the end to prevent importing it from our modules.
