from typing import Any

import openai
from loguru import logger

from reworkd_platform.schemas.workflow.base import Block, BlockIOBase
from reworkd_platform.settings import settings


class CompanyContextAgentInput(BlockIOBase):
    company_name: str


class CompanyContextAgentOutput(CompanyContextAgentInput):
    result: str


class CompanyContextAgent(Block):
    type = "OpenAIAgent"
    description = "Extract key details from text using OpenAI"
    input: CompanyContextAgentInput

    async def run(self, workflow_id: str, **kwargs: Any) -> BlockIOBase:
        try:
            response = await execute_prompt(company=self.input.company_name)

        except Exception as err:
            logger.error(f"Failed to extract text with OpenAI: {err}")
            raise

        return CompanyContextAgentOutput(**self.input.dict(), result=response)


async def execute_prompt(company: str) -> str:
    openai.api_key = settings.openai_api_key
    openai.api_base = settings.openai_api_base

    prompt = f"""
    Write a one-sentence description of "{company}".
    Define their market, sector, and primary products.

    Be as clear, informative, and descriptive as necessary.
    You will not make up information or add any information outside of the above text.
    Only use the given information and nothing more.
    """

    response = openai.ChatCompletion.create(
        model="openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        transforms=["middle-out"],
        headers={ "HTTP-Referer": "http://localhost:3000", "X-Title": "AgentGPT" },
    )

    response_message_content = response["choices"][0]["message"]["content"]

    return response_message_content
