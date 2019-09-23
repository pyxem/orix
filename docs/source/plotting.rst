Plotting
--------

Data visualisation is as important as data handling. In orix, this is still
a work in progress, but several tools are already available. The process builds
upon the way data is plotted in `matplotlib <https://matplotlib.org/>`_.

Any plotting session will begin by importing both the matplotlib plotting API
and the texpy plotting API.

.. ipython::

    In [1]: import matplotlib.pyplot as plt

    In [2]: import texpy.plot

    In [3]: %matplotlib qt5  # Setting the backend.


The next step is to set up a plot. In matplotlib, there is a distinction between
the *figure* - the window for the eventual plot - and the *axes* - the
region of the figure in which the data will be plotted. A figure can contain
any number of axes and each axes object behaves more or
less independently of the others.

.. important::

    The :obj:`~matplotlib.axes.Axes` object is *not* the same as the axes of
    a graph!

An figure and axes can be created in a number of ways (refer to the
`matplotlib <https://matplotlib.org/>`_ documentation). The quickest and
simplest is:

.. ipython::

    In [4]: ax = plt.figure().add_subplot(111)






