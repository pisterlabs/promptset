from datetime import datetime
from functools import wraps
import inspect
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Type,
)

from openai.openai_object import OpenAIObject

from .chat import run_chat
from .config import TurboModel
from .errors import InvalidValueYieldedError
from .memory import LocalMemory

from .structs import (
    Assistant,
    Generate,
    GetInput,
    Result,
    Start,
)

from .structs.proxies import proxy_turbo_gen_fn

from .types import (
    BaseCache,
    BaseMemory,
    BaseMessageCollection,
    Message,
    TurboGenWrapper,
)

from .types.generators import (
    TurboGenFn,
    TurboGenTemplateFn,
)

from .utils.args import ensure_args


__all__ = [
    "turbo",
]


# Decorator
def turbo(
    memory_class: Type[BaseMemory] = LocalMemory,
    model: TurboModel = "gpt-3.5-turbo-0301",
    stream: bool = False,
    cache_class: Optional[Type[BaseCache]] = None,
    debug: Optional[Callable[[dict], None]] = None,
    original_fn: Optional[Callable] = None,
    select_choice: Callable[[List[OpenAIObject]], OpenAIObject] = (
        lambda choices: choices[0]
    ),
    n: int = 1,
    **kwargs,
) -> Callable[[TurboGenTemplateFn], TurboGenFn]:
    """Parameterized decorator for creating a chatml app from an async generator"""

    # Streaming not supported yet
    if stream:
        raise NotImplementedError("Streaming not supported yet")

    # Settings
    settings = dict(
        memory_class=memory_class,
        model=model,
        stream=stream,
        cache_class=cache_class,
        debug=debug,
    )

    # Prepare openai args
    chat_completion_args = {
        **kwargs,
        "model": model,
        "stream": stream,
        "n": n,
    }

    # Parameterized decorator fn
    def wrap_turbo_gen_fn(gen_fn: TurboGenTemplateFn) -> TurboGenFn:
        """Wrapper for chatml app async generator"""

        @wraps(gen_fn)
        async def turbo_gen_fn(*args, **kwargs) -> TurboGenWrapper:
            """Wrapped chatml app from an async generator"""

            # Get cache and memory args
            cache_args = kwargs.pop("cache_args", {})
            memory_args = kwargs.pop("memory_args", {})

            # Prepare kwargs
            signature = inspect.signature(original_fn or gen_fn)
            arg_names = [k for k in signature.parameters.keys()]

            # Init memory
            memory = memory_class(model=model)
            assert ensure_args(memory.setup, memory_args)
            await memory.setup(**memory_args)

            # Init cache
            cache = None
            if cache_class:
                cache = cache_class()
                assert ensure_args(cache.setup, cache_args)
                await cache.setup(**cache_args)

            if "memory" in arg_names:
                kwargs["memory"] = memory

            if "cache" in arg_names:
                kwargs["cache"] = cache

            params = signature.bind(*args, **kwargs)
            context = params.arguments

            # Init generator
            turbo_gen = gen_fn(**context)

            # Parameters
            payload: Any = None

            # Yield Start() to indicate start
            yield Start()

            try:
                while True:
                    # Step through the wrapped generator
                    output = await turbo_gen.asend(payload)

                    # Pass inputs & outputs to debug log if needed
                    if debug:
                        params = dict(
                            app=gen_fn.__name__,
                            timestamp=datetime.utcnow(),
                        )

                        if payload:
                            debug({**params, "type": "input", "payload": payload})

                        if output:
                            debug({**params, "type": "output", "payload": output})

                    payload = None

                    # Add to memory
                    if isinstance(output, Result):
                        output.original_role = "result"
                        yield output

                    elif isinstance(output, Message):
                        await memory.append(output)

                        # Yield to user if forward
                        if output.forward:
                            yield Result.from_message(output)

                    elif isinstance(output, BaseMessageCollection):
                        await memory.extend(await output.get())

                    # Yield to user if GetInput
                    elif isinstance(output, GetInput):
                        if output.record:
                            await memory.append(Assistant(content=output.content))

                        # Get input
                        payload = yield Result.from_message(output)
                        assert payload, f"User input was required, {payload} passed"

                    # Generate result
                    elif isinstance(output, Generate):
                        output_dict = output.dict()
                        forward = output_dict.pop("forward")

                        payload = await run_chat(
                            memory,
                            cache,
                            select_choice=select_choice,
                            **{
                                **chat_completion_args,
                                **output_dict,
                            },
                        )

                        # Yield generated result if needed
                        if forward:
                            yield Result.from_message(payload)

                    else:
                        raise InvalidValueYieldedError(
                            f"Invalid value yielded by generator: {type(output)}"
                        )

            except StopAsyncIteration:
                # Generator over, exit
                pass

        # Add reference to the original function
        turbo_gen_fn = proxy_turbo_gen_fn(turbo_gen_fn)
        turbo_gen_fn.fn = gen_fn
        turbo_gen_fn.settings = settings
        turbo_gen_fn.configure = lambda **new_settings: turbo(
            **{**settings, **new_settings}
        )(gen_fn)

        return turbo_gen_fn

    return wrap_turbo_gen_fn
