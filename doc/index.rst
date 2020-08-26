====
orix
====

.. toctree::
    :hidden:

    reference.rst

.. include:: ../README.rst

Contributing
------------

orix is a community maintained project. We welcome contributions in the form of bug
reports, feature requests, code, documentation, and more. `These guidelines
<https://github.com/pyxem/pyxem/blob/master/CONTRIBUTING.rst>`_ describe how best to
contribute to the `pyxem <https://github.com/pyxem/pyxem>`_ package, but everything
there holds for the orix library as well.

Related projects
----------------

Related, open-source projects that users of orix might find useful:

- `pyxem <https://github.com/pyxem/pyxem>`_: Python library for multi-dimensional
  diffraction microscopy.
- `diffsims <https://github.com/pyxem/diffsims>`_: Python library for simulating
  diffraction.
- `kikuchipy <https://kikuchipy.org>`_: Python library for processing and analysis of
  electron backscatter diffraction (EBSD) patterns.
- `MTEX <https://mtex-toolbox.github.io>`_: Matlab toolbox for analyzing and
  modelling crystallographic textures by means of EBSD or pole figure data.

Work using orix
---------------

.. [Johnstone2020]
    D. N. Johnstone, B. H. Martineau, P. Crout, P. A. Midgley, A. S.
    Eggeman, "Density-based clustering of crystal orientations and
    misorientations and the orix python library," *arXiv
    preprint:2001.02716* (2020), url: https://arxiv.org/abs/2001.02716.

Installation
------------

orix can be installed from `Anaconda <https://anaconda.org/conda-forge/orix>`_, the
`Python Package Index <https://pypi.org/project/orix>`_ (``pip``), or from source, and
supports Python >= 3.6.

We recommend you install it in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
with the `Miniconda distribution`_::

   $ conda create --name orix-env
   $ conda activate orix-env

If you prefer a graphical interface to manage packages and environments, install the
`Anaconda distribution`_ instead.

.. _Miniconda distribution: https://docs.conda.io/en/latest/miniconda.html
.. _Anaconda distribution: https://docs.continuum.io/anaconda/

Anaconda
~~~~~~~~

Anaconda provides the easiest installation. In the Anaconda Prompt, terminal or Command
Prompt, install with::

    $ conda install orix --channel conda-forge

If you at a later time need to update the package::

    $ conda update orix

pip
~~~

To install with ``pip``, run the following in the Anaconda Prompt, terminal or Command
Prompt::

    $ pip install orix

If you at a later time need to update the package::

    $ pip install --upgrade orix

Install from source
~~~~~~~~~~~~~~~~~~~

To install orix from source, clone the repository from `GitHub
<https://github.com/pyxem/orix>`_::

    $ git clone https://github.com/pyxem/orix.git
    $ cd orix
    $ pip install --editable .

Changelog
---------

Changes with each release can be viewed on the `GitHub release page
<https://github.com/pyxem/orix/releases>`_.
