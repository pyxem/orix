#
# Copyright 2018-2026 the orix developers
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

import logging


def set_log_level(level: int | str):  # pragma: no cover
    """Set level of orix logging messages.

    Parameters
    ----------
    level
        Any value accepted by :meth:`logging.Logger.setLevel()`. Levels
        are "DEBUG", "INFO", "WARNING", "ERROR" and "CRITICAL".

    Notes
    -----
    See https://docs.python.org/3/howto/logging.html.

    Examples
    --------
    Note that you might have to set the logging level of the root stream
    handler to display orix' debug messages, as this handler might have
    been initialized by another package

    >>> import logging
    >>> logging.root.handlers[0]  # doctest: +SKIP
    <StreamHandler <stderr> (INFO)>
    >>> logging.root.handlers[0].setLevel("DEBUG")

    >>> import orix.utils as ous
    >>> ous.set_log_level("DEBUG")
    """
    logging.basicConfig()
    logging.getLogger("orix").setLevel(level)
