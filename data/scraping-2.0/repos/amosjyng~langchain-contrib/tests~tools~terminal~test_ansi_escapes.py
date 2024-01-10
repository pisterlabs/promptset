"""Module to interpret terminal output."""

import re
from typing import NamedTuple, Optional

from langchain_contrib.tools.terminal.ansi_escapes import (
    ansi_escape_regex,
    interpret_terminal_output,
)


class EscapeCodeResult(NamedTuple):
    """Result of match on ANSI escape code regex."""

    full_match: str
    escape_code: Optional[str]
    command: Optional[str]


def search(regex: re.Pattern, input: str) -> EscapeCodeResult:
    """Return structured result for escape code regex searches."""
    search_result = re.search(regex, input)
    assert search_result is not None
    escape_code, command = search_result.groups()
    return EscapeCodeResult(
        full_match=search_result.group(),
        escape_code=escape_code,
        command=command,
    )


def test_escape_extraction() -> None:
    """Detect different escape codes and their commands."""
    regex = ansi_escape_regex()
    assert search(regex, "Detect\rthis") == ("\r", None, None)
    assert search(regex, "Detect\\rthis") == ("\\r", None, None)

    assert search(regex, "Detect\033[31;1;4mthis") == (
        "\033[31;1;4m",
        "\033[",
        "31;1;4m",
    )

    assert search(regex, "Detect\\e[31;1;4mthis") == (
        "\\e[31;1;4m",
        "\\e[",
        "31;1;4m",
    )

    assert search(regex, "Detect\x1b[31;1;4mthis") == (
        "\x1b[31;1;4m",
        "\x1b[",
        "31;1;4m",
    )

    assert search(regex, "Detect\u001b[31;1;4mthis") == (
        "\u001b[31;1;4m",
        "\u001b[",
        "31;1;4m",
    )

    assert search(regex, "Detect\x1b[2Kthis") == ("\x1b[2K", "\x1b[", "2K")
    assert search(regex, "Detect\x1b(2Kthis") == ("\x1b(2K", "\x1b(", "2K")


def test_remove_colors() -> None:
    """Remove colors and other formatting."""
    assert interpret_terminal_output("\033[31;1;4mHello\033[0m") == "Hello"
    # Python REPL doesn't show this line right, test with `echo` instead
    assert interpret_terminal_output("\\e[31;1mWorld\\e[39;22m") == "World"
    assert interpret_terminal_output("\x1b[33mbye world...\x1b[39m") == "bye world..."


def test_remove_rs() -> None:
    r"""Treat \r's as erasing everything from the beginning of the line."""
    line = (
        "\r.gitignore            0%[                   ]       0  --.-KB/s               "  # noqa
        "\r.gitignore          100%[===================>]  3.01K  --.-KB/s    in 0s      "  # noqa
    )
    assert (
        interpret_terminal_output(line)
        == ".gitignore          100%[===================>]  3.01K  --.-KB/s    in 0s      "  # noqa
    )


def test_remove_previous_lines() -> None:
    """Remove previously printed lines with A escape code."""
    input = "All that\nEphemerality\n\u001b[1A\u001b[0JRemains"
    assert interpret_terminal_output(input) == "All that\nRemains"


def test_remove_double_digit_lines() -> None:
    """Be able to remove more than 9 previously printed lines at a time."""
    input = b"  \x1b[34;1m\xe2\x80\xa2\x1b[39;22m \x1b[39mInstalling \x1b[39m\x1b[36mfaiss-cpu\x1b[39m\x1b[39m (\x1b[39m\x1b[39;1m1.7.3\x1b[39;22m\x1b[39m)\x1b[39m: \x1b[34mInstalling...\x1b[39m\r\n  \x1b[32;1m\xe2\x80\xa2\x1b[39;22m \x1b[39mInstalling \x1b[39m\x1b[36mflake8\x1b[39m\x1b[39m (\x1b[39m\x1b[32m6.0.0\x1b[39m\x1b[39m)\x1b[39m\r\n  \x1b[32;1m\xe2\x80\xa2\x1b[39;22m \x1b[39mInstalling \x1b[39m\x1b[36mgoogle-search-results\x1b[39m\x1b[39m (\x1b[39m\x1b[32m2.4.1\x1b[39m\x1b[39m)\x1b[39m\r\n  \x1b[32;1m\xe2\x80\xa2\x1b[39;22m \x1b[39mInstalling \x1b[39m\x1b[36mgorilla\x1b[39m\x1b[39m (\x1b[39m\x1b[32m0.4.0\x1b[39m\x1b[39m)\x1b[39m\r\n  \x1b[32;1m\xe2\x80\xa2\x1b[39;22m \x1b[39mInstalling \x1b[39m\x1b[36misort\x1b[39m\x1b[39m (\x1b[39m\x1b[32m5.11.4\x1b[39m\x1b[39m)\x1b[39m\r\n  \x1b[32;1m\xe2\x80\xa2\x1b[39;22m \x1b[39mInstalling \x1b[39m\x1b[36mlangchain\x1b[39m\x1b[39m (\x1b[39m\x1b[32m0.0.100\x1b[39m\x1b[39m)\x1b[39m\r\n  \x1b[32;1m\xe2\x80\xa2\x1b[39;22m \x1b[39mInstalling \x1b[39m\x1b[36mmypy\x1b[39m\x1b[39m (\x1b[39m\x1b[32m0.991\x1b[39m\x1b[39m)\x1b[39m\r\n  \x1b[32;1m\xe2\x80\xa2\x1b[39;22m \x1b[39mInstalling \x1b[39m\x1b[36mopenai\x1b[39m\x1b[39m (\x1b[39m\x1b[32m0.26.4\x1b[39m\x1b[39m)\x1b[39m\r\n  \x1b[32;1m\xe2\x80\xa2\x1b[39;22m \x1b[39mInstalling \x1b[39m\x1b[36mpytest\x1b[39m\x1b[39m (\x1b[39m\x1b[32m7.2.1\x1b[39m\x1b[39m)\x1b[39m\r\n  \x1b[32;1m\xe2\x80\xa2\x1b[39;22m \x1b[39mInstalling \x1b[39m\x1b[36mvcrpy\x1b[39m\x1b[39m (\x1b[39m\x1b[32m4.2.1\x1b[39m\x1b[39m)\x1b[39m\r\n\x1b[10A\x1b[0J  \x1b[32;1m\xe2\x80\xa2\x1b[39;22m \x1b[39mInstalling \x1b[39m\x1b[36mflake8\x1b[39m\x1b[39m (\x1b[39m\x1b[32m6.0.0\x1b[39m\x1b[39m)\x1b[39m\r\n".decode(  # noqa
        "utf-8"
    )
    assert interpret_terminal_output(input) == "  • Installing flake8 (6.0.0)\n"


def test_remove_b() -> None:
    """Remove B escape codes."""
    # Pyhon REPL doesn't remove all B's -- try with `echo` instead to test
    input = "\\e[1m\\e[32mSuccess: no issues found in 3 source files\\e(B\\e[m"
    assert (
        interpret_terminal_output(input) == "Success: no issues found in 3 source files"
    )


def test_remove_k() -> None:
    """Remove K escape codes in general."""
    input = "\033[2KResolving dependencies...\033[2KResolving dependencies..."
    assert interpret_terminal_output(input) == "Resolving dependencies..."


def test_remove_k_no_number() -> None:
    """Remove K escape codes with no numerical argument."""
    # Python REPL doesn't remove all K's -- try with `echo` instead to test
    input = "remote: Resolving deltas:   0% (0/3)\\e[K\rremote: Resolving deltas:  33% (1/3)\\e[K\rremote: Resolving deltas:  66% (2/3)\\e[K\rremote: Resolving deltas: 100% (3/3)\\e[K\rremote: Resolving deltas: 100% (3/3), completed with 3 local objects.\\e[K\r\nremote: \r\nremote: Create a pull request for 'upgrade/langchain-v0.0.100'"  # noqa
    assert (
        interpret_terminal_output(input)
        == """
remote: Resolving deltas: 100% (3/3), completed with 3 local objects.
remote: 
remote: Create a pull request for 'upgrade/langchain-v0.0.100'
""".strip()  # noqa
    )
