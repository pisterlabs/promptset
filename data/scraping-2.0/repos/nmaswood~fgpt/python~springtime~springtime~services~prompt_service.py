import abc
import time

import anthropic
import pydantic
from anthropic import Anthropic
from loguru import logger

from springtime.services.html import html_from_text


class PromptRequest(pydantic.BaseModel):
    template: str
    args: dict[str, str]


class PromptResponse(pydantic.BaseModel):
    raw: str
    html: str | None
    input_tokens: pydantic.NonNegativeInt
    output_tokens: pydantic.NonNegativeInt
    prompt: str


class PromptService(abc.ABC):
    @abc.abstractmethod
    def run(
        self,
        req: PromptRequest,
    ) -> PromptResponse:
        pass


class PromptServiceImpl(
    PromptService,
):
    def __init__(self, anthropic: Anthropic) -> None:
        self.anthropic = anthropic

    def run(
        self,
        req: PromptRequest,
    ) -> PromptResponse:
        prompt = """
Human: {template}



Assistant:
""".format(
            template=req.template,
        )

        prompt = prompt.format(**req.args)

        input_tokens = self.anthropic.count_tokens(prompt)

        def get_response() -> str:
            for attempt in range(9):
                try:
                    return self.anthropic.completions.create(
                        model="claude-2",
                        max_tokens_to_sample=1_000_000,
                        prompt=prompt,
                    ).completion.strip()
                except anthropic.RateLimitError:
                    seconds = 2 ** (attempt + 2)
                    logger.info(f"Rate limit exceeded sleeping {seconds}")

                    time.sleep(seconds)
            msg = "Rate limit exceeded"
            raise Exception(msg)

        response = get_response()

        output_tokens = self.anthropic.count_tokens(response)
        html = html_from_text(response)

        return PromptResponse(
            raw=response,
            html=html,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt=prompt,
        )
