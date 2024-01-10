import asyncio
import textwrap
import time
from typing import AsyncGenerator, Generator

from clonr.tokenizer import Tokenizer

from .callbacks import LLMCallback
from .openai import OpenAI, OpenAIStreamResponse
from .schemas import (
    FinishReason,
    GenerationParams,
    LLMResponse,
    Message,
    RoleEnum,
    StreamChoice,
    StreamDelta,
    Usage,
)


class MockLLM(OpenAI):
    model_type: str = "mock"
    _counter: int = 0
    is_chat_model: bool = True
    model: str = "gpt-3.5-turbo"

    def __init__(
        self,
        response: str = "mock response",
        callbacks: list[LLMCallback] | None = None,
    ):
        self.response = response
        self.tokenizer = Tokenizer.from_openai("gpt-3.5-turbo")
        self.callbacks = callbacks or []

    @property
    def user_start(self):
        return "<|im_start|>user\n"

    @property
    def user_end(self):
        return "<|im_end|>"

    @property
    def assistant_start(self):
        return "<|im_start|>assistant\n"

    @property
    def assistant_end(self):
        return self.user_end

    @property
    def system_start(self):
        return "<|im_start|>system\n"

    @property
    def system_end(self):
        return self.user_end

    @property
    def default_system_prompt(self):
        return "You are a helpful assistant."

    @property
    def context_length(self) -> int:
        return 4000

    def generate(self, *args, **kwargs) -> LLMResponse:
        return asyncio.get_event_loop().run_until_complete(  # type: ignore
            self.generate(*args, **kwargs)
        )

    async def agenerate(
        self,
        prompt_or_messages: str | list[Message],
        params: GenerationParams | None = None,
        **kwargs,
    ) -> LLMResponse:
        st = time.time()
        if params is None:
            params = GenerationParams()
        self._counter += 1
        content = self.response
        content = self.tokenizer.decode(
            self.tokenizer.encode(content)[: params.max_tokens]
        )
        prompt = prompt_or_messages
        if not isinstance(prompt, str):
            prompt = " ".join([x.content for x in prompt])

        for c in self.callbacks:
            await c.on_generate_start(self, prompt, params, **kwargs)

        if isinstance(prompt_or_messages, str):
            prompt_tokens = self.num_tokens(self.prompt_to_messages(prompt_or_messages))
        else:
            prompt_tokens = self.num_tokens(prompt_or_messages)
        completion_tokens = self.tokenizer.length(content)
        total_tokens = prompt_tokens + completion_tokens
        usage = Usage(
            total_tokens=total_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
        )
        r = LLMResponse(
            content=content,
            model_type=self.model_type,
            model_name="MockModel",
            created_at=round(time.time()),
            usage=usage,
            finish_reason=FinishReason.length,
            role="assistant",
            duration=1 + time.time() - st,
            input_prompt=prompt,
            tokens_per_second=1.0,
        )

        for c in self.callbacks:
            await c.on_generate_end(self, r, **kwargs)

        return r

    async def astream(
        self, *args, **kwargs
    ) -> AsyncGenerator[OpenAIStreamResponse, None]:
        # (Jonny): no callbacks for the mock streaming stuff, because I'm too lazy.
        for x in self.stream():
            yield x

    def stream(self, *args, **kwargs) -> Generator[OpenAIStreamResponse, None, None]:
        self._counter += 1
        for ch in "ABC":
            yield OpenAIStreamResponse(
                created=round(time.time()),
                model="MockModel",
                choices=[
                    StreamChoice(
                        index=0,
                        delta=StreamDelta(
                            role=RoleEnum.user,
                            content=f" ResponseToken{self._counter}{ch}",
                        ),
                        finish_reason=FinishReason.length if ch == "C" else None,
                    )
                ],
            )

    async def notebook_stream_(self, *args, **kwargs) -> list[OpenAIStreamResponse]:
        from IPython import display

        r = []
        text = ""
        async for x in self.astream():
            text += x.choices[0].delta.content or ""
            _text = "\n".join(textwrap.wrap(text, width=70))
            display.clear_output(wait=False)
            print(_text, flush=True)
            r.append(x)
        return r
