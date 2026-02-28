Code style
==========

The code making up orix is formatted closely following the `Style Guide for Python Code
<https://www.python.org/dev/peps/pep-0008/>`__ with :doc:`The Black Code style
<black:the_black_code_style/current_style>` and `isort
<https://pycqa.github.io/isort/>`__ to handle module imports.
We use `pre-commit <https://pre-commit.com>`__ to run ``black`` and ``isort``
automatically prior to each local commit.
Please install it in your environment:

.. code-block:: bash

    pre-commit install

Next time you commit some code, your code will be formatted inplace according to the
default black configuration.

Note that ``black`` won't format `docstrings <https://peps.python.org/pep-0257/>`__.
We follow the :doc:`numpydoc <numpydoc:format>` standard (with some exceptions), and
docstrings are checked against this standard when the documentation is built.

Package imports should be structured into three blocks with blank lines between them
(descending order): standard library (like ``os`` and ``typing``), third party packages
(like ``numpy`` and ``matplotlib``) and finally first party ``orix`` imports.
``isort`` will structure the import order in this way by default.
Note that if imports must be sorted in a certain order, for example to avoid recursion,
then ``isort`` provides `commands
<https://pycqa.github.io/isort/docs/configuration/action_comments.html>`__ that may be
used to prevent sorting.

Comment and docstring lines should preferably be limited to 72 characters (including
leading whitespaces).

We use `type hints <https://docs.python.org/3/library/typing.html>`__` in the function
signature without type duplication in the docstring, for example:

.. code-block:: python

    def my_function(arg1: int, arg2: bool | None = None) -> tuple[float, np.ndarray]:
        """This is a new function.

        Parameters
        ----------
        arg1
            Explanation about argument 1.
        arg2
            Explanation about flag argument 2. Default is None.

        Returns
        -------
        values
            Explanation about returned values.
        """

For new code, please use ``tuple[]``, ``list[]``, ``dict[]``, etc. instead of the
deprecated capitalized variants.
Instead of ``Optional[bool]``, use ``bool | None``.
Instead of ``Union[int, float]``, use ``int | float``.

When working with classes in orix, often a method argument will require another instance
of the class.
An example of this is :meth:`orix.vector.Vector3d.dot`, where the first argument to this
function ``other`` is another instance of ``Vector3d``.
In this case, to allow for the correct type hinting behaviour, the following import is
required at the top of the file:

.. code-block:: python

    from __future__ import annotations

Type hints for various built-in classes are available from the ``typing`` module.
``np.ndarray`` should be used for arrays.

Mathematical notation
---------------------

See :ref:`mathematical_notation`.
