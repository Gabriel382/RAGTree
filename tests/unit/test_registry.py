"""Unit tests for the component registry."""

import pytest

from ragtree.core.registry import build, register


def test_register_and_build_roundtrip():
    @register("unit-test.widget", "simple")
    class Simple:
        def __init__(self, value=0):
            self.value = value

    obj = build("unit-test.widget", "simple", value=3)
    assert isinstance(obj, Simple)
    assert obj.value == 3


def test_build_unknown_kind_raises_key_error():
    with pytest.raises(KeyError):
        build("unit-test.nothing", "missing")


def test_build_unknown_name_raises_key_error():
    @register("unit-test.gadget", "known")
    class Known:
        pass

    with pytest.raises(KeyError):
        build("unit-test.gadget", "unknown")
