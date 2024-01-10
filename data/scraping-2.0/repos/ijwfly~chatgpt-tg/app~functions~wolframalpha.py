import re
from typing import Optional

import settings
from app.functions.base import OpenAIFunction, OpenAIFunctionParams

from pydantic import Field
import httpx


FIELDS_TO_EXTRACT = ['Input interpretation', 'Result', 'Results']


class QueryWolframAlphaParams(OpenAIFunctionParams):
    query: str = Field(..., description="query for WolframAlpha (translated to english, if needed)")


class QueryWolframAlpha(OpenAIFunction):
    PARAMS_SCHEMA = QueryWolframAlphaParams

    @staticmethod
    async def query_wolframalpha(query: str):
        url = 'https://www.wolframalpha.com/api/v1/llm-api'
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params={
                'appid': settings.WOLFRAMALPHA_APPID,
                'input': query,
            })
        if resp.status_code != 200:
            raise Exception(f'WolframAlpha returned {resp.status_code} status code with message: {resp.text}')

        if not 'Result' in resp.text and not 'Results' in resp.text:
            return resp.text

        results = []
        for field in FIELDS_TO_EXTRACT:
            pattern = r'({}:\n.+?)\n\n'.format(field)
            results += re.findall(pattern, resp.text, re.DOTALL)

        return '\n'.join(results)

    async def run(self, params: QueryWolframAlphaParams) -> Optional[str]:
        try:
            return await self.query_wolframalpha(params.query)
        except Exception as e:
            return f"Error: {e}"

    @classmethod
    def get_name(cls) -> str:
        return "query_wolframalpha"

    @classmethod
    def get_description(cls) -> str:
        return "Query WolframAlpha for factual info and calculations"
