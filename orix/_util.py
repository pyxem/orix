# -*- coding: utf-8 -*-
# Copyright 2018-2021 the orix developers
#
# This file is part of orix.
#
# orix is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# orix is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix.  If not, see <http://www.gnu.org/licenses/>.

"""Helper functions and classes for managing orix.

This module and documentation is only relevant for orix developers, not
for users.

.. warning:
    This module and its submodules are for internal use only.  Do not
    use them in your own code. We may change the API at any time with no
    warning.
"""


import functools
import inspect
import warnings

import numpy as np


class deprecated:
    """Decorator to mark deprecated functions or properties with an
    informative warning.

    Adapted from
    `scikit-image
    <https://github.com/scikit-image/scikit-image/blob/main/skimage/_shared/utils.py#L297>`_
    and `matplotlib
    <https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/_api/deprecation.py#L122>`_.
    """

    def __init__(self, since, alternative=None, removal=None, object_type="function"):
        """Visible deprecation warning.

        Parameters
        ----------
        since : str, int or float
            The release at which this API became deprecated.
        alternative : str, optional
            An alternative API that the user may use in place of the
            deprecated API.
        removal : str, int or float, optional
            The expected removal version.
        object_type : str, optional
            Type of the deprecated object, either "function" or
            "property".
        """
        self.since = since
        self.alternative = alternative
        self.removal = removal
        self.object_type = object_type

    def __call__(self, func):
        # Wrap function or property to raise warning when called, and
        # add warning to docstring
        if self.object_type.lower() != "function":
            object_type = "Property"
            parentheses = ""
        else:
            object_type = "Function"
            parentheses = "()"
        if self.alternative is not None:
            alt_msg = f" Use `{self.alternative}{parentheses}` instead."
        else:
            alt_msg = ""
        if self.removal is not None:
            rm_msg = f" and will be removed in version {self.removal}"
        else:
            rm_msg = ""
        msg = (
            f"{object_type} `{func.__name__}{parentheses}` is deprecated"
            f"{rm_msg}.{alt_msg}"
        )

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.simplefilter(
                action="always", category=np.VisibleDeprecationWarning
            )
            func_code = func.__code__
            warnings.warn_explicit(
                message=msg,
                category=np.VisibleDeprecationWarning,
                filename=func_code.co_filename,
                lineno=func_code.co_firstlineno + 1,
            )
            return func(*args, **kwargs)

        # Modify docstring to display deprecation warning
        old_doc = inspect.cleandoc(func.__doc__ or "").strip("\n")
        notes_header = "\nNotes\n-----"
        new_doc = (
            f"[*Deprecated*] {old_doc}\n"
            f"{notes_header if notes_header not in old_doc else ''}\n"
            f".. deprecated:: {self.since}\n"
            f"   {msg.strip()}"  # Matplotlib uses three spaces
        )
        wrapped.__doc__ = new_doc

        return wrapped
