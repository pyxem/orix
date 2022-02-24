# -*- coding: utf-8 -*-
# Copyright 2018-2022 the orix developers
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

import numpy as np
import pytest

from orix._util import deprecated


class TestDeprecationWarning:
    def test_deprecation_since(self):
        """Ensure functions decorated with the custom deprecated
        decorator returns desired output, raises a desired warning, and
        gets the desired additions to their docstring.
        """

        class Foo:
            @property
            @deprecated(
                since="1.3",
                alternative="bar_prop2",
                removal="1.4",
                object_type="property",
            )
            def bar_prop(self):
                return 1

            @deprecated(since=0.7, alternative="bar_func3", removal=0.8)
            def bar_func1(self, n):
                """Some docstring."""
                return n + 1

            @deprecated(since=1.9)
            def bar_func2(self, n):
                """Another docstring.

                Notes
                -----
                Some existing notes.
                """
                return n + 2

            @deprecated(
                since="1.3",
                alternative="other_kwarg",
                removal="1.4",
                object_type="argument",
                argument="kwarg1",
            )
            def bar_arg(self, other_kwarg=2):
                return other_kwarg

            @deprecated(
                since="1.3",
                alternative="other_kwarg",
                removal="1.4",
                object_type="argument",
                argument="kwarg1",
                extra="This is the extra",
            )
            def bar_arg_extra(self, other_kwarg=2):
                return other_kwarg

            with pytest.raises(
                TypeError, match="argument must be defined for object_type"
            ):
                # fmt: off
                @deprecated(
                    since="1.3",
                    alternative="other_kwarg",
                    removal="1.4",
                    object_type="argument",
                    argument=None,
                )
                def bar_arg_no_arg(self, other_kwarg=2): pass
                # fmt: on

        my_foo = Foo()

        with pytest.warns(np.VisibleDeprecationWarning) as record:
            assert my_foo.bar_func1(4) == 5
        desired_msg = (
            "Function `bar_func1()` is deprecated and will be removed in version 0.8. "
            "Use `bar_func3()` instead."
        )
        assert str(record[0].message) == desired_msg
        assert my_foo.bar_func1.__doc__ == (
            "[*Deprecated*] Some docstring.\n"
            "\nNotes\n-----\n"
            ".. deprecated:: 0.7\n"
            f"   {desired_msg}"
        )

        with pytest.warns(np.VisibleDeprecationWarning) as record2:
            assert my_foo.bar_func2(4) == 6
        desired_msg2 = "Function `bar_func2()` is deprecated."
        assert str(record2[0].message) == desired_msg2
        assert my_foo.bar_func2.__doc__ == (
            "[*Deprecated*] Another docstring.\n"
            "\nNotes\n-----\n"
            "Some existing notes.\n\n"
            ".. deprecated:: 1.9\n"
            f"   {desired_msg2}"
        )

        with pytest.warns(np.VisibleDeprecationWarning) as record3:
            assert my_foo.bar_prop == 1
        desired_msg3 = (
            "Property `bar_prop` is deprecated and will be removed in version 1.4. "
            "Use `bar_prop2` instead."
        )
        assert str(record3[0].message) == desired_msg3
        assert my_foo.__class__.bar_prop.__doc__ == (
            "[*Deprecated*] \n"
            "\nNotes\n-----\n"
            ".. deprecated:: 1.3\n"
            f"   {desired_msg3}"
        )

        with pytest.warns(np.VisibleDeprecationWarning) as record4:
            assert my_foo.bar_arg(kwarg1=3) == 2
        desired_msg4 = (
            "Argument `kwarg1` is deprecated and will be removed in version 1.4. "
            "Use `other_kwarg` instead."
        )
        assert str(record4[0].message) == desired_msg4

        with pytest.warns(np.VisibleDeprecationWarning) as record5:
            assert my_foo.bar_arg_extra(kwarg1=3) == 2
        desired_msg5 = (
            "Argument `kwarg1` is deprecated and will be removed in version 1.4. "
            "Use `other_kwarg` instead. This is the extra."
        )
        assert str(record5[0].message) == desired_msg5

        # test incorrect object_type
        with pytest.raises(ValueError, match="object_type must be one of"):

            @deprecated("1.4", object_type="not possible")
            # fmt: off
            def foo1(): pass
            # fmt: on

            foo1()
