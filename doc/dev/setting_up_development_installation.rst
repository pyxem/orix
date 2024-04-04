.. _setting-up-a-development-installation:

Setting up a development installation
=====================================

You need a `fork
<https://docs.github.com/en/get-started/quickstart/contributing-to-projects#about-forking>`__
of the `repository <https://github.com/pyxem/orix>`__ in order to make changes to orix.

Make a local copy of your forked repository and change directories::

    git clone https://github.com/your-username/orix.git
    cd orix

Set the ``upstream`` remote to the main orix repository::

    git remote add upstream https://github.com/pyxem/orix.git

We recommend installing in a `conda environment
<https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
with the `Miniconda distribution <https://docs.conda.io/en/latest/miniconda.html>`__::

    conda create --name orix-dev
    conda activate orix-dev

Then, install the required dependencies while making the development version available
globally (in the ``conda`` environment)::

    pip install --editable ".[dev]"

This installs all necessary development dependencies, including those for running tests
and building documentation.
