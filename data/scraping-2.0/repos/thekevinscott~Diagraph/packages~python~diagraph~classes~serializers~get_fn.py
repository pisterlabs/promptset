# from types import CodeType, FunctionType
import re
from collections.abc import Callable
from inspect import getsource as _getsource
from textwrap import dedent
from typing import Any

from ...decorators.prompt import prompt
from ...llm.openai_llm import OpenAI
from ...utils.depends import Depends
from ..types import Fn

default_vars = {"Depends": Depends, "prompt": prompt, "Fn": Fn, "OpenAI": OpenAI}


def get_fn_name(func_str: str) -> str:
    return func_str.split("def ")[1].split("(")[0]


def get_fn(
    func_str: str,
    fn_name: None | str = None,
    node_args: dict[str, Any] | None = None,
):
    local_vars = {
        **(node_args or {}),
    }

    # Use exec to execute the function string in the context of the empty dictionary
    node_args = {
        **default_vars,
    }
    exec(
        func_str,
        {
            **globals(),
            **node_args,
        },
        local_vars,
    )

    if fn_name is None:
        fn_name = get_fn_name(func_str)

    fn = local_vars.get(fn_name)
    if fn is None:
        raise ValueError(
            f'Invalid function string, could not find function for name "{fn_name}". Local vars: {local_vars}',
        )

    if callable(fn):
        return fn
    raise ValueError("Invalid function string")


def dump_fn(fn: Callable) -> str:
    fn_str = dedent(_getsource(fn)).strip()

    def rewrite_depends(s: re.Match[str]) -> str:
        arg = s.group(1).strip("\"'")
        return f'Depends("{arg}")'

    return re.sub(r"Depends\((.*?)\)", rewrite_depends, fn_str)
