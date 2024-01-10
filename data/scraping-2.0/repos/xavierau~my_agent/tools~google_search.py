# import os
import json
import os
from typing import List, Optional

import tiktoken
from langchain.text_splitter import CharacterTextSplitter
import dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from utils.llm import get_response_content_from_gpt
from utils.logger import Logger
from googleapiclient.discovery import build
import asyncio
from pyppeteer import launch
from bs4 import BeautifulSoup

dotenv.load_dotenv()

from tools.common import Tool, ToolCallResult


class SummarizeTool(BaseModel):
    text: str
    requirement: Optional[str] = None
    model_name: str = Field(default="gpt-3.5-turbo-1106")

    async def get_summary(self, prompt: str = None) -> str:
        client = AsyncOpenAI()

        messages = [
            {
                "role": "user",
                "content": self.default_prompt if prompt is None else prompt
            }
        ]

        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        return response.choices[0].message.content

    @property
    def default_prompt(self):
        prompt = f"""Please summarize the text for the purpose below:

                    Text:
                    {self.text}
                    `````

                    %purpose%

                    Summary:"""

        if self.requirement is not None:
            return prompt.replace('%purpose%', f"""Purpose:
                        {self.requirement}
                        `````
                        """)

        return prompt.replace('%purpose%', '')


class MapReduceSummarizeTool(SummarizeTool):
    text: str
    requirement: Optional[str] = None
    model_name: str = Field(default="gpt-3.5-turbo-1106")
    chunk_size: int = Field(default=2048)
    overlap: int = Field(default=100)

    async def get_summary(self) -> str:
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.overlap
        )
        texts = text_splitter.split_text(self.text)

        summaries = await asyncio.gather(
            *[SummarizeTool(text=text, requirement=self.requirement).get_summary() for text in texts])

        summaries = '\n'.join(summaries)

        suggestion = f"""Base on the following summaries, answer user's question.
                                               Summaries:
                                               {summaries}
                                               `````
                                               %requirement%

                                               Answer:"""

        if self.requirement:
            suggestion = suggestion.replace('%requirement%', f"""User Question:
                                               {self.requirement}
                                               `````
                    """)
        else:
            suggestion = suggestion.replace('%requirement%', '')

        return await SummarizeTool(text="").get_summary(suggestion)


class GoogleSearchTool(Tool):
    """Search from Google"""
    name: str = "google_search"
    description: str = "It is helpful when you need to search information from internet"
    summary_model: str = "gpt-3.5-turbo"

    @property
    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query you want to search. Take into account about the user preferences.",
                        },
                        "limit": {
                            "type": "number",
                            "description": "Number of top result return. It must between 1 and 10. Without special reason, always set it to 3.",
                            "default": 3
                        }
                    },
                    "required": ["query", "limit"]
                }
            }
        }

    key: str
    search_engine_id: str

    async def run(self, query: str, limit=5) -> ToolCallResult:

        Logger.info(f"tool:{self.name} query: {query}, limit: {limit}")

        print('search query: ', query)

        if query is None:
            raise Exception

        items = await self._get_result_from_google_search(limit, query)

        summarized_websites = await asyncio.gather(*[self._summarize_website(query, item['link']) for item in items])

        results = []
        for idx, item in enumerate(items):
            results.append(self._to_structure(query, item, summarized_websites[idx]))

        return ToolCallResult(result=json.dumps(results))

    async def _get_result_from_google_search(self, limit, query):
        service = build(
            "customsearch", "v1",
            developerKey=os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
        )
        res = (
            service.cse()
            .list(
                q=query,
                cx=os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
            )
            .execute()
        )
        items = res.get('items', [])
        if len(items) > limit:
            items = items[:limit]
        return items

    def _to_structure(self, question: str, data: dict, summary):
        return {
            "site_title": data.get("title"),
            "source": data.get("link"),
            "summary": summary
        }

    async def _summarize_site(self, question: str, content: str) -> str | None:

        try:
            encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
            number_of_tokens = len(encoder.encode(content))
            print("token count: ", number_of_tokens)
            if number_of_tokens < 2048:
                return await SummarizeTool(text=content, requirement=question).get_summary()
            else:
                return await MapReduceSummarizeTool(text=content, requirement=question).get_summary()

        except Exception as e:
            print("Something wrong about fetching the url")
            print(e)
            return None

    async def _fetch_page_content(self, url):
        print("try url: ", url)
        try:
            # Launch the browser
            browser = await launch()
            page = await browser.newPage()

            # Navigate to the URL
            await page.goto(url)
            print("visiting url", url)

            # Wait for the page to load (you can customize this)
            await asyncio.sleep(2)  # Waits for 2 seconds

            print("finished waiting", url)

            # Get page content after JavaScript is loaded
            content = await page.content()

            print("get content", url)

            # Use BeautifulSoup to parse the content
            soup = BeautifulSoup(content, 'html.parser')

            # Extract data
            page_text = soup.get_text()

            print("parsed content", url)

            await browser.close()
            return page_text

        except Exception as e:
            print(f"Something went wrong: {e}", url)
            return None

    async def _summarize_website(self, query: str, url: str):

        page_content = await self._fetch_page_content(url)

        return await self._summarize_site(query, page_content)
