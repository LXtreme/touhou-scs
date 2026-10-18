"""
Pytest bootstrap.

touhou_scs requires init_level() before anything else is imported (unknown_g()
is called at import time in lib.py), and lib must be imported before component
to avoid a circular import. Real levels need a GD file/editor, so tests bind the
allocator to a stub that hands out sequential group IDs.
"""

import itertools
import types

from touhou_scs.utils import init_level

_group_counter = itertools.count(5000)

init_level(types.SimpleNamespace(  # type: ignore[arg-type]
    new=types.SimpleNamespace(
        group=lambda: next(_group_counter),
        used_group_ids=set(),
    )
))

from touhou_scs import lib  # type: ignore noqa: E402,F401  (must come before component)

import pytest  # noqa: E402


@pytest.fixture
def fake_level():
    """Stand-in for gmdbuilder Level, for lib.save_all(level=fake_level, live_editor=False)."""
    return types.SimpleNamespace(
        objects=[],
        export_to_file=lambda: None,
        export_to_live_editor=lambda: None,
    )
