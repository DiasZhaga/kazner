"""Deliberately failing test: demonstrates that a red CI run blocks the merge (#2)."""


def test_ci_blocks_merge_demo():
    assert 1 + 1 == 3, "deliberate failure to demonstrate merge blocking"
