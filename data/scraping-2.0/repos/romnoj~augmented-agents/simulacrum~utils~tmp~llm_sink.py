from datetime import datetime
from typing import List, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

from langchain.llms.base import Generation, LLMResult, BaseLLM

import subprocess
import asyncio


class ChatParameters(BaseModel):
    model: str = Field(default="vicuna")
    temperature: float = Field(default=0.2)

    top_k: int = Field(default=50)
    top_p: float = Field(default=0.95)

    max_length: int = Field(default=256)

    repeat_last_n: int = Field(default=64)
    repeat_penalty: float = Field(default=1.3)


class Question(BaseModel):
    question: str
    answer: str


class Chat(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    created: datetime = Field(default_factory=datetime.now)
    questions: Optional[List[Question]]
    parameters: ChatParameters


def remove_matching_end(a, b):
    min_length = min(len(a), len(b))

    for i in range(min_length, 0, -1):
        if a[-i:] == b[:i]:
            return b[i:]

    return b


async def generate(
    model: str = "vicuna",
    prompt: str = "One of the most beautiful things is..",
    n_predict: int = 300,
    temp: float = 0.8,
    top_k: int = 10000,
    top_p: float = 0.40,
    repeat_last_n: int = 100,
    repeat_penalty: float = 1.2,
    chunk_size: int = 4,  # chunk size (in bytes) for streaming
) -> AsyncGenerator[str, None]:
    args = (
        # "/home/ubuntu/repos/augmented-agents/llama.cpp/models/wizard-vicuna-13B.ggmlv3.q4_0.bin",
        "--model",
        "" + model,
        "--prompt",
        prompt,
        "--n_predict",
        str(n_predict),
        "--temp",
        str(temp),
        "--top_k",
        str(top_k),
        "--top_p",
        str(top_p),
        "--repeat_last_n",
        str(repeat_last_n),
        "--repeat_penalty",
        str(repeat_penalty),
        "--threads",
        "8",
    )
    print(args)
    procLlama = await asyncio.create_subprocess_exec(
        *args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    answer = ""
    while True:
        chunk = await procLlama.stdout.read(chunk_size)
        if not chunk:
            return_code = await procLlama.wait()

            if return_code != 0:
                error_output = await procLlama.stderr.read()
                raise ValueError(error_output.decode("utf-8"))
            else:
                return

        chunk = chunk.decode("utf-8")
        print(chunk, end="", flush=True)
        answer += chunk

        if prompt in answer:
            yield remove_matching_end(prompt, chunk)


class Llama(BaseLLM, BaseModel):
    async def _agenerate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        response = ""
        async for token in generate(prompt=prompts[0]):
            response += token
            self.callback_manager.on_llm_new_token(token, verbose=True)

        generations = [[Generation(text=response)]]
        return LLMResult(generations=generations)

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._agenerate(prompts, stop))
        finally:
            loop.close()

        return result

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        result = self._generate([prompt], stop)
        return result.generations[0][0].text

    @property
    def _identifying_params(self) -> dict:
        return {}

    @property
    def _llm_type(self) -> str:
        return "llama"
