# Copyright (c) 2023 Rocket Science AG, Switzerland

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Contains the logic for interpolables, i.e. functions that can be called from within a message, like
{user_input()} or {python()}.

The basic idea is that a user can enter a message like

    ```
    {user_input()}
    ```

The message is then parsed and the code block is extracted. In this case, the code block is

    {user_input()}

The code block is then checked against all interpolables to see which one is responsible for it. In this case, the
`UserInputInterpolable` would be responsible. The interpolable is then executed and the output is returned. In this
case, the user would be asked for input.
"""

import asyncio
import io
import re
from abc import ABC, abstractmethod
from contextlib import redirect_stdout
from typing import Awaitable, Callable

import attrs
import pandas as pd
from overrides import override

from rrosti.chat.chat_session import Importance
from rrosti.llm_api import openai_api
from rrosti.servers.websocket_query_server import Frontend, PythonItem
from rrosti.snippets import document_sync
from rrosti.snippets.abstract_snippet_database import AbstractSnippetDatabase
from rrosti.snippets.sklearn_snippet_database import SklearnSnippetDatabase
from rrosti.snippets.snippet import Snippet
from rrosti.utils import misc
from rrosti.utils.config import config


class NoCodeBlockFoundError(ValueError):
    pass


@attrs.frozen
class InterpolableOutput:
    output: str

    # For debug purposes, shown in frontend. If None, then the actual output is not shown at all.
    # If it should be shown but without prefix, simply use ''
    info_prefix: str | None

    importance: Importance  # Influences pruning order

    # None = do not prune; positive integer = prune after this many user inputs
    ttl: int | None = attrs.field()

    @ttl.validator
    def _check_ttl(self, _attribute: str, value: int | None) -> None:
        if value is not None and value <= 0:
            raise ValueError("ttl must be None or a positive integer")


# TODO: Is this name descriptive? Can we think of something better?
class Interpolable(ABC):
    """An interpolable function invocation, like {user_input()}, {python()} or {rtfm()}."""

    @abstractmethod
    def is_for_me(self, code: str) -> bool:
        ...

    @abstractmethod
    async def execute(self, last_msg: str | None, frontend: Frontend) -> InterpolableOutput:
        ...


# A simple interpolable that wraps a coroutine
class SimpleInterpolable(Interpolable):
    _code: str
    _coro: Callable[[], Awaitable[str]]
    _importance: Importance
    _ttl: int | None

    def __init__(self, code: str, importance: Importance, ttl: int | None, coro: Callable[[], Awaitable[str]]) -> None:
        self._code = code
        self._coro = coro
        self._importance = importance
        self._ttl = ttl

    @override
    def is_for_me(self, code: str) -> bool:
        return code == self._code

    @override
    async def execute(self, last_msg: str | None, frontend: Frontend) -> InterpolableOutput:
        return InterpolableOutput(
            output=await self._coro(), info_prefix=None, importance=self._importance, ttl=self._ttl
        )


def extract_code_block(language: str, msg_text: str) -> str:
    # First check that we don't have multiple matches for ```{language}
    if msg_text.count(f"```{language}") > 1:
        raise ValueError("Multiple code blocks found in message.")

    code_block_pattern = re.compile(r"\$\$\$" + language + r"\r?\n(.*?)\$\$\$", re.DOTALL)
    match = code_block_pattern.search(msg_text)
    if not match:
        raise NoCodeBlockFoundError("No full and terminated code block found in message")
    return match.group(1)


def execute_python_in_msg(msg_text: str, vars: dict[str, object]) -> PythonItem:
    code = extract_code_block("python", msg_text)

    try:
        f = io.StringIO()
        with redirect_stdout(f):
            exec(code, dict(pd=pd, **vars))
        out = f.getvalue().strip()
    except Exception as e:
        out = "Exception: " + str(e)
    return PythonItem(_code=code, _output=out)


@misc.async_once_blocking
async def _get_database() -> AbstractSnippetDatabase:
    snippets = await document_sync.sync_and_get_snippets()
    return await asyncio.to_thread(SklearnSnippetDatabase, snippets)


async def ensure_loaded() -> None:
    await _get_database()


async def execute_rtfm_in_msg(openai_provider: openai_api.OpenAIApiProvider, message: str) -> list[Snippet]:
    text = extract_code_block("rtfm", message)

    # We need the encoding for the text
    snippet = Snippet.from_query(text)
    await snippet.async_ensure_embedding(openai_provider)

    # Now execute the nearest neighbors search
    db = await _get_database()
    return await db.find_nearest_merged(
        openai_provider, snippet, config.state_machine.rtfm_max_tokens, config.state_machine.rtfm_merge_candidates
    )
