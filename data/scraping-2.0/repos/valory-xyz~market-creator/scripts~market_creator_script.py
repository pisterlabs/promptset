# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2023 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""Script for creating market creator."""

import datetime
import os
import random
import time
from typing import Any, Dict, Optional, Tuple

import openai
import requests


DEFAULT_OPENAI_SETTINGS = {
    "max_tokens": 500,
    "temperature": 0.7,
}

TOOL_TO_ENGINE = {
    # "market-creator": "gpt-3.5-turbo",
    "market-creator": "gpt-4",
}

ALLOWED_TOOLS = list(TOOL_TO_ENGINE.keys())

MARKET_CREATION_PROMPT = """
Based on the following news snippets under "INPUT", formulate 5 prediction market questions with clear, objective
outcomes that can be verified on specific dates and leave no room for interpretation or subjectivity.
Avoid incorporating questions that could potentially encourage unethical behavior or violence.
Every question should be resolvable on  of August 2023.
Your questions should follow a structure similar to this:
"Will VERIFIABLE_BINARY_OUTCOME_OR_PREDICTION occur on SPECIFIC_DAY_OF_AUGUST_2023".
Your output must follow the output format detailed under "OUTPUT FORMAT".

INPUT
{input_news}

OUTPUT_FORMAT
* Your output response must be only a single JSON array to be parsed by Python's "json.loads()".
* All of the date strings should be represented in YYYY-MM-DD format.
* Each entry of the JSON array must be a JSON object containing the fields:
    - question: The binary question to open a prediction market.
    - answers: The possible answers to the question.
    - resolution_date: The resolution date for the outcome of the market to be verified.
    - topic: One word description of the topic of the news and it should be one of: {topics}.
* Output only the JSON object. Do not include any other contents in your response.
"""

TOPICS = '["business","science","technology","politics","arts","weather"]'


def run(  # pylint: disable=too-many-locals
    **kwargs,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Run the task"""
    openai.api_key = kwargs["api_keys"]["openai"]
    newsapi_api_key = kwargs["api_keys"]["newsapi"]
    max_tokens = kwargs.get("max_tokens", DEFAULT_OPENAI_SETTINGS["max_tokens"])
    temperature = kwargs.get("temperature", DEFAULT_OPENAI_SETTINGS["temperature"])
    tool = kwargs["tool"]

    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {tool} is not supported.")

    engine = TOOL_TO_ENGINE[tool]

    newsapi_url = "https://newsapi.org/v2/everything"

    newsapi_headers = {"X-Api-Key": newsapi_api_key}

    today = datetime.date.today()
    from_date = today - datetime.timedelta(days=7)
    to_date = today

    params = {
        "q": "arts OR business OR finance OR cryptocurrency OR politics OR science OR technology OR sports OR weather OR entertainment",
        "language": "en",
        "sortBy": "popularity",
        "from": from_date,
        "to": to_date,
    }

    response = requests.get(newsapi_url, params=params, headers=newsapi_headers)
    data = response.json()

    print(data)

    # Create the string with the desired format
    articles = data["articles"]
    random.shuffle(articles)
    articles = articles[:20]

    input_news = ""
    for article in articles:
        title = article["title"]
        content = article["content"]
        date = article["publishedAt"]
        input_news += f"- ({date}) {title}\n  {content}\n\n"

    market_creation_prompt = MARKET_CREATION_PROMPT.format(
        input_news=input_news, from_date=from_date, to_date=to_date, topics=TOPICS
    )

    print(market_creation_prompt)

    start_time = time.time()

    moderation_result = openai.Moderation.create(market_creation_prompt)

    if moderation_result["results"][0]["flagged"]:
        return "Moderation flagged the prompt as in violation of terms."

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": market_creation_prompt},
    ]

    response = openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        timeout=120,
        stop=None,
    )

    end_time = time.time()

    print(response.choices[0].message.content)

    elapsed_time = end_time - start_time

    print(f"Function took {elapsed_time:.2f} seconds to execute.")

    return response.choices[0].message.content, None


# Testing the script
run(
    **{
        "prompt": "unused",
        "tool": "market-creator",
        "api_keys": {
            "openai": os.environ.get("OPENAI_API_KEY"),
            "newsapi": os.environ.get("NEWSAPI_API_KEY"),
        },
    }
)
