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

import orix.quaternion as oqu
import orix.utils as ous


class TestLogging:
    def test_logging(self, caplog):
        ori = oqu.Orientation.random(10)

        logger = logging.getLogger("orix")
        ous.set_log_level("WARNING")
        assert logger.level == 30

        _ = ori.mean()

        # No info messages are logged
        assert len(caplog.records) == 0

        ous.set_log_level("INFO")
        assert logger.level == 20

        # Info messages are logged
        _ = ori.mean()
        assert len(caplog.records) > 0
        for record in caplog.records:
            assert record.levelname == "INFO"
