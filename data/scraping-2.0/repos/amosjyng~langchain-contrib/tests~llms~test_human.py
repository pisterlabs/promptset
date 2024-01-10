"""Tests for the human LLM."""

from unittest.mock import patch

from langchain_contrib.llms.human import Human


def test_simple_input() -> None:
    """Test that we can get some simple input from the human."""
    with patch("builtins.input", return_value="World"):
        llm = Human()
        assert llm("Hello") == "World"


def test_multiline_input() -> None:
    """Test that we can get some multiline input from the human."""
    with patch(
        "builtins.input", side_effect=["#!/usr/bin/env python3", "print('Hi')", "```"]
    ):
        llm = Human()
        assert (
            llm("Write me a Python script", stop=["```"])
            == """
#!/usr/bin/env python3
print('Hi')
""".lstrip()
        )


def test_multiline_stop() -> None:
    """Test that the stop can involve newlines."""
    with patch(
        "builtins.input",
        side_effect=[
            "#!/usr/bin/env python3",
            "print('Hi')",
            "print('Bye')",
            "```",
        ],
    ):
        llm = Human()
        assert (
            llm("Write me a Python script\n\n```\n", stop=["\n```"])
            == """
#!/usr/bin/env python3
print('Hi')
print('Bye')""".lstrip()
        )
