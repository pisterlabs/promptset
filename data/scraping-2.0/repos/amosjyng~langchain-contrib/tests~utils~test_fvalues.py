"""Module to test enhanced fvalues functionality."""

from fvalues import F, FValue

from langchain_contrib.utils.fvalues import f_join


def test_f_join_empty() -> None:
    """Test f_join with an empty list."""
    s = f_join(" ", [])
    assert s == F("", parts=("",))


def test_f_join_list() -> None:
    """Test f_join with a list of strings."""
    b = 2
    s = f_join(" ", ["a", F(f"b={b}"), "c"])
    assert s == "a b=2 c"
    assert s.parts == (
        "a",
        " ",
        F(f"b={b}", parts=("b=", FValue(source="b", value=2, formatted="2"))),
        " ",
        "c",
    )
    assert s.flatten().parts == (
        "a",
        " ",
        "b=",
        FValue(source="b", value=2, formatted="2"),
        " ",
        "c",
    )


def test_f_join_empty_string() -> None:
    """Test f_join with an empty joiner string."""
    b = 2
    s = f_join("", ["a", F(f"b={b}"), "c"])
    assert s == "ab=2c"
    assert s.parts == (
        "a",
        F(f"b={b}", parts=("b=", FValue(source="b", value=2, formatted="2"))),
        "c",
    )
    assert s.flatten().parts == (
        "a",
        "b=",
        FValue(source="b", value=2, formatted="2"),
        "c",
    )
