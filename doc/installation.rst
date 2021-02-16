============
Installation
============

orix can be installed from `Anaconda <https://anaconda.org/conda-forge/orix>`_, the
`Python Package Index <https://pypi.org/project/orix/>`_ (``pip``), or from source, and
supports Python >= 3.6.

We recommend you install it in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
with the `Miniconda distribution`_::

   $ conda create --name orix-env python=3.8
   $ conda activate orix-env

If you prefer a graphical interface to manage packages and environments, install the
`Anaconda distribution`_ instead.

.. _Miniconda distribution: https://docs.conda.io/en/latest/miniconda.html
.. _Anaconda distribution: https://docs.continuum.io/anaconda/

.. _install-with-anaconda:

Anaconda
--------

Anaconda provides the easiest installation. In the Anaconda Prompt, terminal or Command
Prompt, install with::

    $ conda install orix --channel conda-forge

If you at a later time need to update the package::

    $ conda update orix

.. _install-with-pip:

Pip
---

To install with ``pip``, run the following in the Anaconda Prompt, terminal or Command
Prompt::

    $ pip install orix

If you at a later time need to update the package::

    $ pip install --upgrade orix

.. _install-from-source:

Install from source
-------------------

To install orix from source, clone the repository from `GitHub
<https://github.com/pyxem/orix>`_::

    $ git clone https://github.com/pyxem/orix.git
    $ cd orix
    $ pip install --editable .

See the :ref:`contributing guidelines <set-up-a-development-installation>`
for how to set up a development installation.
