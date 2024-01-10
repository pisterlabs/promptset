from typing import Annotated

from fastapi import Depends
from langchain.llms import BaseLLM, OpenAI

from ...config import Settings, get_settings
from ...llms.cron.cron import CronExpressionGenerator


def get_openai_instance(settings: Annotated[Settings, Depends(get_settings)]) -> OpenAI:
    return OpenAI(
        openai_api_key=settings.openai_api_key,
        temperature=0.0,
        max_tokens=150,
        model_name="text-davinci-003",
    )


def get_cron_expression_generator(
    llm: Annotated[BaseLLM, Depends(get_openai_instance)],
) -> CronExpressionGenerator:
    return CronExpressionGenerator(llm)
