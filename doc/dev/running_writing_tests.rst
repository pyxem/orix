Run and write tests
===================

All functionality in orix is tested with :doc:`pytest <pytest:index>`.
The tests reside in a ``tests`` module.
Tests are short methods that call functions in ``orix`` and compare resulting output
values with known answers.
Install necessary dependencies to run the tests::

   pip install --editable ".[tests]"

Some useful :doc:`fixtures <pytest:explanation/fixtures>` are available in the
``conftest.py`` file.

.. note::

    Some :mod:`orix.data` module tests check that data not part of the package
    distribution can be downloaded from the web, thus downloading some small datasets to
    your local cache.
    See the section on the :ref:`data module <adding-data-to-data-module>` for more
    details.

To run the tests::

   pytest --cov --pyargs orix

The ``--cov`` flag makes :doc:`coverage.py <coverage:index>` print a nice report.
For an even nicer presentation, you can use ``coverage.py`` directly::

   coverage html

Coverage can then be inspected in the browser by opening ``htmlcov/index.html``.

We strive for 100% test coverage of lines when all dependencies are installed.

Docstring examples are tested with :doc:`pytest <pytest:how-to/doctest>` as well.
:mod:`numpy` and :mod:`matplotlib.pyplot` should not be imported in examples as they are
already available in the namespace as ``np`` and ``plt``, respectively.
The docstring tests can be run from the top directory::

    pytest orix --doctest-modules --ignore-glob=orix/tests

Tips for writing tests of Numba decorated functions:

- A Numba decorated function ``numba_func()`` is only covered if it is called in the
  test as ``numba_func.py_func()``.
- Always test a Numba decorated function calling ``numba_func()`` directly, in addition
  to ``numba_func.py_func()``, because the machine code function might give different
  results on different OS with the same Python code.
