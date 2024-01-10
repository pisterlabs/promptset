import os
from typing import Awaitable, Callable, Literal, Optional, TypeVar

import attrs
import openai
import tiktoken
from tqdm.asyncio import tqdm_asyncio

from monitoring.json_cache import json_converter
from rrutils.llm_api.base_llm import LLMResponse
from rrutils.llm_api.openai_llm import OpenAIChatModel
from rrutils.redis_cache_wrapper import RedisWrapper, cache_key


@attrs.frozen
class ChatInput:
    prompt: tuple[tuple[str, str], ...]
    temp: float = 0.0
    n: int = 1


CachedResponses = dict[ChatInput, list[LLMResponse]]


@attrs.frozen
class WrappedCache:
    resps: list[tuple[ChatInput, list[LLMResponse]]]

    @classmethod
    def from_normal(cls, x: CachedResponses):
        return cls([(k, v) for k, v in x.items()])

    def to_normal(self) -> CachedResponses:
        return {k: v for k, v in self.resps}


@attrs.define
class CachedOpenAIChatModel:
    model: OpenAIChatModel
    model_ids: set[str]

    async def __call__(self, inp: ChatInput, max_tokens_HACK: Optional[int] = None):
        # Max tokens not cached, currently this is a hack!
        key = cache_key(json_converter.dumps(inp), f"model_cache:{str(sorted(self.model_ids))}")

        cached_res = await RedisWrapper.singleton().read(key)
        if cached_res is not None:
            cached_res = [json_converter.structure(x, LLMResponse) for x in cached_res]
            return cached_res, True

        kwargs = {}
        if max_tokens_HACK is not None:
            kwargs["max_tokens"] = max_tokens_HACK
        out = await self.model(
            list(self.model_ids),
            [{"role": role, "content": content} for role, content in inp.prompt],
            print_prompt_and_response=False,
            max_attempts=10,
            temperature=inp.temp,
            n=inp.n,
            **kwargs,
        )
        await RedisWrapper.singleton().write(key, [json_converter.unstructure(x) for x in out])
        return out, False


Role = Literal["user", "system", "assistant"]

tokenizer = tiktoken.encoding_for_model("gpt-4")

if "OPENAI_API_KEY" not in os.environ:
    openai.api_key_path = os.path.expanduser("~/.openai_api_key")
    try:
        with open(openai.api_key_path, "r") as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()
    except FileNotFoundError:
        ...


smart_models_strs = {"gpt-4", "gpt-4-0613"}
dumb_models_strs = {"gpt-3.5-turbo", "gpt-3.5-turbo-0613"}
dumb_models_long_strs = {"gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"}
smart_models = CachedOpenAIChatModel(OpenAIChatModel(0.95), smart_models_strs)
dumb_models = CachedOpenAIChatModel(OpenAIChatModel(0.95), dumb_models_strs)
dumb_models_long = CachedOpenAIChatModel(OpenAIChatModel(0.95), dumb_models_long_strs)

T = TypeVar("T")
U = TypeVar("U")


async def general_get_completions_with_price(
    items: list[T], call: Callable[[T], Awaitable[tuple[U, list[LLMResponse], bool]]], progress: bool = True
) -> list[U]:
    async def run_item(x: T):
        out, items, cached = await call(x)
        return out, sum(x.cost for x in items), cached

    out_all = await tqdm_asyncio.gather(*[run_item(x) for x in items], disable=not progress)
    out = [x[0] for x in out_all]
    price = sum(x[1] for x in out_all)
    price_cached = sum(x[1] for x in out_all if not x[2])

    if not progress:
        print(f"Price: ${price:.4f} Price cached: ${price_cached:.4f}")
    return out
