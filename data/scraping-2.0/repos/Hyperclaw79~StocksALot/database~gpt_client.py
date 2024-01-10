"""
This module contains the GptClient client which will interact with the ChatGPT API.
"""
from __future__ import annotations
import json
import os
import re
import openai
from utils import logger_factory, dedent
from models import Message, GptRoles, InsightsResponse

MODEL = os.getenv("GPT_MODEL", "gpt-4")
logger = logger_factory(__name__)


class GptClient:  # pylint: disable=too-few-public-methods
    """
    A client for interacting with the OpenAI GPT API.
    """
    insights_pattern = re.compile(
        r'(\s\(Sentiment:\s(Positive|Neutral|Negative)\))',
        flags=re.IGNORECASE
    )

    def __init__(self, api_key: str, model: str = MODEL):
        openai.api_key = api_key
        self.model = model
        self.last_prompted_datetime = None
        # pylint: disable=line-too-long
        self.behavior_instruction = dedent("""You are a stock market expert, capable of quickly analysing trends and outliers in stock data.
            You are given a list of stock data and you need to provide a minimum of 3 and a maximum of 5 insights per datetime.

            Some rules for generating the insights:
            1. Insights should be easy enough event for a 10th grade kid to understand.
            2. Rank them according to this key: (steepness of the change, impact on market).
            3. Provide a sentiment (positive, negative, neutral) for each insight.
            4. Each unique datetime should have a corresponsing list of insights.
            5. If multiple datetimes are provided, create the insight for the newer date based on shifts in trends from the older date.
            6. Do not try to guess trends if a single datetime is provided. Just provide insights for that datetime.
            7. Maintain uniqueness; avoid repeating same insight for different stocks.
            8. Insights should sound creative and human-like.
            9. Try to get a mix of positive, neutral and negative insights if possible.
            10. **IMPORTANT** - Any figures you quoute should exactly match the figures in the data."""
        )
        self.cached_insights: InsightsResponse = None

    async def prompt(self, stock_data: list[dict]) -> InsightsResponse:
        """Sends a prompt to the OpenAI GPT API and returns the response."""
        requested_datetime = max(stock_data, key=lambda x: x["datetime"])["datetime"]
        if self.last_prompted_datetime == requested_datetime:
            return self.cached_insights
        prompt_str = f"""
        For the following stock data, get a minimum of 3 and a maximum of 5 insights per datetime.
        ```json
        {json.dumps(stock_data, default=str)}
        ```
        """
        messages = [
            Message(role=GptRoles.SYSTEM, content=self.behavior_instruction),
            Message(role=GptRoles.USER, content=prompt_str)
        ]
        logger.info("Sending prompt to GPT API for insights.")
        try:
            response = await self._send_prompt(messages)
            self.clean_insights(response["items"])
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to get insights from GPT API.")
            logger.error(exc)
            return InsightsResponse(count=0, items=[])
        self.last_prompted_datetime = requested_datetime
        self.cached_insights = InsightsResponse(**response)
        return InsightsResponse(**response)

    async def _send_prompt(self, messages: list[Message]) -> dict[
        str, int | list[dict[str, str | list[str]]]
    ]:
        """Sends a prompt to the OpenAI GPT API and returns the response."""
        func_response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[msg.model_dump() for msg in messages],
            functions=[{
                "name": "get_stock_insights",
                "description": "Get a minimum of 3 and a maximum of 5 insights"
                " from the provided stock data.",
                "parameters": InsightsResponse.model_json_schema()
            }],
            function_call={"name": "get_stock_insights"}
        )
        response = json.loads(
            func_response.choices[0].message
            .function_call.arguments
        )
        logger.info("Received JSON response from GPT API for insights.")
        if isinstance(response, list):
            response = {"count": len(response), "items": response}
        response["count"] = len(response["items"])
        for item in response["items"]:
            del item["insights"][5:]
        return response

    @classmethod
    def clean_insights(cls, insights: list[dict[str, str | list[dict[str, str]]]]):
        """Cleans the insights by removing the sentiment and other noise."""
        for record in insights:
            for insight in record["insights"]:
                message = insight.pop("message", None) or insight.pop("insight", None)
                insight["message"] = cls.insights_pattern.sub("", message)
