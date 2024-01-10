"""Lazy objects for xontrib_chatgpt"""
import re
from collections import namedtuple

#############
# Lazy Objects
#############


def _openai():
    """Imports openai"""
    import openai

    return openai


def _tiktoken():
    """Imports tiktoken"""
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _MULTI_LINE_CODE():
    """Regex to remove multiline code blocks (```code```) from markdown"""
    return re.compile(r"```.*?\n", re.DOTALL)


def _SINGLE_LINE_CODE():
    """Regex to remove single line code blocks (`code`) from markdown"""
    return re.compile(r"`(.*?)`")


def _PYGMENTS():
    """Lazy loading of pygments to avoid slowing down shell startup"""
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.lexers.python import PythonLexer
    from pygments.formatters import Terminal256Formatter
    from pygments.styles.gh_dark import GhDarkStyle

    container = namedtuple(
        "container",
        [
            "highlight",
            "get_lexer_by_name",
            "PythonLexer",
            "Terminal256Formatter",
            "GhDarkStyle",
        ],
    )
    return container(
        highlight, get_lexer_by_name, PythonLexer, Terminal256Formatter, GhDarkStyle
    )


def _markdown():
    """Formats markdown text using pygments"""
    from pygments import highlight
    from pygments.lexers.markup import MarkdownLexer
    from pygments.formatters import Terminal256Formatter
    from pygments.styles.gh_dark import GhDarkStyle

    return lambda text: highlight(
        text, MarkdownLexer(), Terminal256Formatter(style=GhDarkStyle)
    )


def _FIND_NAME_REGEX():
    """Regex to find the name of a chat from a file name"""
    return re.compile(r"^(?:.+?_)*([a-zA-Z0-9]+)(?:_[0-9\-]+)?\.(?:.*)$", re.DOTALL)

def _YAML():
    """Imports yaml package if it exists"""
    from importlib.util import find_spec

    if find_spec("yaml") is not None:
        import yaml
        return yaml
    else:
        return None