Installation
------------


Requirements
============

orix is written in Python 3. The Anaconda installation instructions below
will take care of this but if you wish to install some other way ensure
Python 3 is the default version.

orix depends heavily on `numpy <http://www.numpy.org/>`_, and requires
`matplotlib <https://matplotlib.org/>`_ for creating figures and images.
One or two functions also require `scipy <https://scipy.org/>`_.
These will be installed automatically alongside orix.

To build the documentation, you will need to manually install
`Sphinx <http://www.sphinx-doc.org/en/stable/index.html>`_,
`sphinx_rtd_theme <https://sphinx-rtd-theme.readthedocs.io/en/latest/>`_,
and `autodocsumm <http://autodocsumm.readthedocs.io/en/latest/?badge=latest>`_.

To run tests, install `pytest <https://docs.pytest.org/en/latest/>`_.


Installation via Anaconda (Recommended)
=======================================

As orix is still under development there has been no official release,
but installation via GitHub is easy. It is recommended you install using
`Anaconda <https://www.anaconda.com/download/>`_.

1. Download and install
   `Anaconda <https://www.anaconda.com/download/>`__ according to system
   instructions.
2. Open a terminal (In Windows, open the start menu and search for
   "Command Prompt") and navigate a suitable local directory. Create and
   activate a clean environment for orix:

   .. code:: shell

      > conda create --name orix-env python=3
      > activate orixenv  # Windows
      > source activate orix-env  # Linux/MacOS

   .. note:: Windows Powershell users may have to run the command "cmd" before
      activating the environment, as virtual environments appear to be
      unsupported in Powershell.

3. Install orix from GitHub using pip:

   .. code:: shell

      > pip install git+https://github.com/pyxem/orix.git

   This will always install the latest version, alongside numpy and matplotlib.


Installation from source
========================

1. Visit `the source code on GitHub <https://github.com/pyxem/orix>`_ and
   download the zip file or simply click
   `here <https://github.com/pyxem/orix/archive/master.zip>`_ to download
   texpy as a zip file.
2. Unzip the downloaded file to a convenient local directory.
3. Open a terminal and navigate to the top level "orix" directory.
4. Ensure you are using Python version 3.5 or higher:

   .. code:: shell

      > python --version

   If not, consider upgrading your python version, or start a virtual
   environment with Python 3.5 or higher.

5. Install texpy:

   .. code:: shell

      > pip install .

   This will install texpy alongside numpy and matplotlib.



