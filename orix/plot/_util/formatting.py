#
# Copyright 2018-2025 the orix developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix. If not, see <http://www.gnu.org/licenses/>.
#

"""Private utilities for formatting text and such in visualization."""

import numpy as np


def format_labels(
    v: np.ndarray | list | tuple,
    brackets: tuple[str, str] | None = None,
    use_latex: bool = True,
) -> np.ndarray:
    r"""Return formatted vector integer labels.

    This function is a convenient way to get nice labels when plotting
    vectors in the stereographic projection via
    :meth:`~orix.vector.Vector3d.scatter` or
    :meth:`~orix.plot.StereographicPlot.text`.

    Parameters
    ----------
    v
        Vector labels with the last dimension having a size of 3 or 4.
        The labels are rounded to the closets integers before being
        formatted.
    brackets
        Left and right parentheses as a tuple of strings. Default is
        ("(", ")"). These are typically () or [] when labeling
        reciprocal (hkl) or direct [uvw] lattice vectors, respectively.
        If not given, no parentheses are added. If *use_latex* is True,
        the brackets {} and <> will be escaped if given.
    use_latex
        Whether to use LaTeX when formatting the labels. Default is
        True.

    Returns
    -------
    new_labels
        Array of string labels.

    Examples
    --------
    >>> from orix.plot import format_labels
    >>> from orix.vector import Vector3d
    >>> v = Vector3d([[1, 1, 1], [-2, 0, 1], [4, 0, 0], [-4, 0, 0]])
    >>> format_labels(v.reshape(2, 2).data)
    array([['$111$', '$\\bar{2}01$'],
           ['$400$', '$\\bar{4}00$']], dtype='<U11')
    >>> format_labels(v.data, ("[", "]"), use_latex=False).tolist()
    ['[111]', '[-201]', '[400]', '[-400]']
    >>> format_labels(v.data, ("{", "}")).tolist()
    ['$\\{111\\}$', '$\\{\\bar{2}01\\}$', '$\\{400\\}$', '$\\{\\bar{4}00\\}$']
    >>> format_labels(v.data, ("{", "}"), use_latex=False).tolist()
    ['{111}', '{-201}', '{400}', '{-400}']
    >>> format_labels(v.data, ("<", ">")).tolist()
    ['$\\left<111\\right>$',
     '$\\left<\\bar{2}01\\right>$',
     '$\\left<400\\right>$',
     '$\\left<\\bar{4}00\\right>$']
    """
    if brackets is None:
        brackets = ("", "")

    if use_latex:
        start = end = r"$"
        if brackets == ("<", ">"):
            brackets = (r"\left<", r"\right>")
        elif brackets == ("{", "}"):
            brackets = (r"\{", r"\}")
    else:
        start = end = ""

    v = np.asanyarray(v)
    shape = v.shape
    v = v.round().astype(int).reshape(-1, shape[-1])

    new_labels = []
    for label in v:
        new_label = start + brackets[0]
        for i in label:
            if i < 0 and use_latex:
                new_label += r"\bar{" + str(abs(i)) + r"}"
            else:
                new_label += str(i)
        new_label += brackets[1] + end
        new_labels.append(new_label)
    new_labels = np.asarray(new_labels).reshape(shape[:-1])

    return new_labels
