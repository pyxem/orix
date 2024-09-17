# Copyright 2018-2024 the orix developers
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

"""Helper functions and classes for managing orix."""


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

    def __init__(
        self,
        since,
        removal=None,
        object_type="function",
        alternative=None,
    ):
        """Visible deprecation warning.

        Parameters
        ----------
        since : str, int or float
            The release at which this API became deprecated.
        removal : str, int or float, optional
            The expected removal version.
        object_type : str, optional
            Type of the deprecated object, either "function" (default)
            or "property".
        alternative : str, optional
            An alternative API that the user may use in place of the
            deprecated API.
        """
        self.since = since
        self.alternative = alternative
        self.removal = removal
        self.object_type = object_type

    def __call__(self, func):
        # Wrap function or property to raise warning when called, and
        # add warning to docstring if function or property is deprecated

        object_type = self.object_type.lower()
        if object_type == "function":
            parentheses = "()"
        else:
            parentheses = ""
        if self.alternative is not None:
            alt_msg = f" Use `{self.alternative}{parentheses}` instead."
        else:
            alt_msg = ""
        if self.removal is not None:
            rm_msg = f" and will be removed in version {self.removal}"
        else:
            rm_msg = ""
        msg = (
            f"{object_type.capitalize()} `{func.__name__}{parentheses}` is deprecated"
            f"{rm_msg}.{alt_msg}"
        )

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.simplefilter(
                action="always", category=np.VisibleDeprecationWarning, append=True
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


class deprecated_argument:
    """Decorator to remove an argument from a function or method's
    signature.

    Adopted from
    `scikit-image
    <https://github.com/scikit-image/scikit-image/blob/main/skimage/_shared/utils.py#L115>`_.
    """

    def __init__(self, name, since, removal, alternative=None):
        self.name = name
        self.since = since
        self.removal = removal
        self.alternative = alternative

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if self.name in kwargs.keys():
                msg = (
                    f"Argument `{self.name}` is deprecated and will be removed in "
                    f"version {self.removal}. To avoid this warning, please do not use "
                    f"`{self.name}`. "
                )
                if self.alternative is not None:
                    msg += f"Use `{self.alternative}` instead. "
                msg += f"See the documentation of `{func.__name__}()` for more details."
                warnings.simplefilter(
                    action="always", category=np.VisibleDeprecationWarning, append=True
                )
                func_code = func.__code__
                warnings.warn_explicit(
                    message=msg,
                    category=np.VisibleDeprecationWarning,
                    filename=func_code.co_filename,
                    lineno=func_code.co_firstlineno + 1,
                )
            return func(*args, **kwargs)

        return wrapped
