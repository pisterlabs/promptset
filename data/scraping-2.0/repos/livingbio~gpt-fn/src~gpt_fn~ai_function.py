from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

import openai

from .completion import chat_completion
from .exceptions import AiFnError
from .prompt import ChatTemplate, MessageTemplate
from .utils.signature import FunctionSignature

T = TypeVar("T")
P = ParamSpec("P")


def ai_fn(model: str = "gpt-3.5-turbo") -> Callable[[Callable[P, T]], Callable[P, T]]:
    def _ai_fn(
        fn: Callable[P, T],
    ) -> Callable[P, T]:
        sig = FunctionSignature(fn)

        @wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> T:
            fn_call = sig.call_line(*args, **kwargs)
            template = ChatTemplate(
                messages=[
                    MessageTemplate(role="system", content=sig.instruction()),
                    MessageTemplate(role="user", content=fn_call),
                ]
            )

            try:
                resp = chat_completion(template.render(), temperature=0.0, model=model)
            except openai.APIError as e:
                fn_locals = sig.locals(*args, **kwargs)
                raise AiFnError(fn_call, fn_locals) from e

            return sig.parse(resp)

        return inner

    return _ai_fn
