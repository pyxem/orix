Build and write documentation
=============================

The documentation contains three categories of documents: ``examples``, ``tutorials``
and the ``reference``.
The documentation strategy is based on the
`Diátaxis Framework <https://diataxis.fr/>`__.
New documents should fit into one of these categories.

We use :doc:`Sphinx <sphinx:index>` for documenting functionality.
Install necessary dependencies to build the documentation::

    pip install --editable ".[doc]"

.. note::

    The tutorials and examples require some small datasets to be downloaded via the
    :mod:`orix.data` module upon building the documentation.
    See the section on the :ref:`data module <adding-data-to-data-module>` for more
    details.

Then, build the documentation from the ``doc`` directory::

    cd doc
    make html

The documentation's HTML pages are built in the ``doc/build/html`` directory from files
in the :doc:`reStructuredText (reST) <sphinx:usage/restructuredtext/basics>` plaintext
markup language.
They should be accessible in the browser by typing
``file:///your-absolute/path/to/orix/doc/build/html/index.html`` in the address bar.

We can link to other documentation in reStructuredText files using
:doc:`Intersphinx <sphinx:usage/extensions/intersphinx>`.
Which links are available from a package's documentation can be obtained like so::

    python -m sphinx.ext.intersphinx https://hyperspy.org/hyperspy-doc/current/objects.inv

We use :doc:`Sphinx-Gallery <sphinx-gallery:index>` to build the :ref:`examples`.
The examples are located in the top source directory ``examples/``, and a new directory
``doc/examples/`` is created when the docs are built.

We use :doc:`nbsphinx <nbsphinx:index>` for converting notebooks into tutorials.
Code lines in notebooks should be :ref:`formatted with black <code-style>`.

Writing tutorial notebooks
--------------------------

Here are some tips for writing tutorial notebooks:

- Notebooks (with the ``.ipynb`` file extension) are ignored by git (listed in the
  ``.gitignore`` file). The ``-f`` `git flag
  <https://git-scm.com/docs/git-add#Documentation/git-add.txt--f>`_ must be added to
  ``git add -f notebook.ipynb`` in order to update an existing notebook or add a new
  one. Notebooks are ignored by git in general to avoid non-documentation changes to
  notebooks, like cell IDs, being pushed unnecessarily.
- All notebooks should have a Markdown (MD) cell with this message at the top,
  "This notebook is part of the ``orix`` documentation https://orix.readthedocs.io.
  Links to the documentation won't work from the notebook.", and have
  ``"nbsphinx": "hidden"`` in the cell metadata so that the message is not visible when
  displayed in the documentation.
- Use ``ax[0].imshow(...);`` to silence ``matplotlib`` output if a ``matplotlib``
  command is the last line in a cell.
- Refer to our API reference with this MD
  ``[Vector3d.zvector()](../reference/generated/orix.vector.Vector3d.zvector.rst)``.
  Remember to add the parentheses ``()`` if the reference points to a function or
  method.
- Refer to to the examples section with ``[Examples section](../examples/index.rst)``.
- Refer to sections in other tutorial notebooks using this MD
  ``[plotting](../tutorials/crystal_map.ipynb#Plotting)``.
- Refer to external APIs via standard MD like
  ``[Lattice](https://www.diffpy.org/diffpy.structure/mod_lattice.html#diffpy.structure.lattice.Lattice)``.
- The Sphinx gallery thumbnail used for a notebook is set by adding the
  ``nbsphinx-thumbnail`` tag to a code cell with an image output. The notebook must be
  added to the gallery in the relevant topic within the user guide to be included in the
  documentation pages.
- The ``furo`` Sphinx theme displays the documentation in a light or dark theme,
  depending on the browser/OS setting. It is important to make sure the documentation is
  readable with both themes. This means for example displaying all figures with a white
  background for axes labels and ticks and figure titles etc. to be readable.
- Whenever the documentation is built (locally or on the Read the Docs server),
  ``nbsphinx`` only runs the notebooks *without* any cell output stored. It is
  recommended that notebooks are stored without cell output, so that functionality
  within them are run and tested to ensure continued compatibility with code changes.
  Cell output should only be stored in notebooks which are too computationally intensive
  for the Read the Docs server to handle, which has a limit of 15 minutes and 3 GB of
  memory per :doc:`documentation build <readthedocs:builds>`.
- We also use ``black`` to format notebooks cells. To run the ``black`` formatter on
  your notebook(s) locally please specify the notebook(s), ie.
  ``black my_notebook.ipynb`` or ``black *.ipynb``, as ``black .`` will not format
  ``.ipynb`` files without explicit consent. To prevent ``black`` from automatically
  formatting regions of your code, please wrap these code blocks with the following::

      # fmt: off
      python_code_block = not_to_be_formatted
      # fmt: on

  Please see the :doc:`black documentation <black:index>` for more details.

In general, we run all notebooks every time the documentation is built with Sphinx, to
ensure that all notebooks are compatible with the current API at all times.
This is important!
For computationally expensive notebooks however, we store the cell outputs so the
documentation doesn't take too long to build, either by us locally or the Read The Docs
GitHub action.
To check that the notebooks with stored cell outputs are compatible with the current
API, we run a scheduled GitHub Action every Monday morning which checks that the
notebooks run OK and that they produce the same output now as when they were last
executed.
We use :doc:`nbval <nbval:index>` for this.

The tutorial notebooks can be run interactively in the browser with the help of Binder.
When creating a server from the orix source code, Binder installs the packages listed in
the ``environment.yml`` configuration file, which must include all ``doc`` dependencies
in ``pyproject.toml`` necessary to run the notebooks.

Writing API reference
---------------------

Inherited attributes and methods are not listed in the API reference unless they are
explicitly coded in the inheriting class.

A class' ``set()`` method, if it has any, is excluded from the API reference.
This is necessary because some plotting classes inheriting from Matplotlib's ``Axes()``
class caused errors when the inherited ``set()`` method is to be included in the API
reference by Sphinx (even though inherited methods are also explicitly excluded).

.. _mathematical_notation:

Mathematical notation
---------------------

We try to use a mathematical notation consistent throughout our documentation and
internal (not user-facing) source code for naming variables.
Rotation objects are denoted by uppercase letters and vector objects are denoted by
lowercase letters.

* Quaternion :math:`Q = (a, b, c, d)`
* Rotation axis :math:`\hat{\mathbf{n}}`
* Rotation angle :math:`\omega`
* Rotation :math:`R = (a, b, c, d)`
* Orientation :math:`O = (a, b, c, d)`
* Misorientation :math:`M = (a, b, c, d)`
* Symmetry operations:

  * Set of operations :math:`S`
  * Single operation :math:`s = (a, b, c, d)`

* 3D vector :math:`\mathbf{v} = (x, y, z)`
* Reciprocal or direct lattice vectors (``Miller``) with coordinate formats:

  * "xyz": :math:`\mathbf{m} = (x, y, z)`
  * "hkl" or "hkil": :math:`\mathbf{g} = (h, k, l)`
  * "uvw" or "UVTW": :math:`\mathbf{t} = [u, v, w]`

* Polar angles

  * Azimuth :math:`\phi`
  * Polar :math:`\theta`

* Stereographic coordinates :math:`(X, Y)`
