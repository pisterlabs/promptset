"""
Wrapper around raw completion API.

Need separate document for abstractions over a raw document
"""

import asyncio
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


class Document:
    def __init__(self, seed: Optional[int] = None):
        self.exchange_template = "user: {{USER}}\n\nbot:"
        self.document = ""
        self.seed = seed
        self.client = AsyncOpenAI()

    async def prompt(self, text: str):
        self.document = self.document + self.exchange_template.replace("{{USER}}", text)
        stream = await self.client.completions.create(
            model="text-davinci-003", prompt=self.document, stream=True, seed=self.seed, max_tokens=1000
        )
        parts = []
        async for completion in stream:
            parts.append(completion.choices[0].text)
            print(completion.choices[0].text, end="")
        self.document += "\n" + "".join(parts) + "\n"
        return self.document


async def main() -> None:
    document = Document()
    await document.prompt("Summarize the first episode of Loki, the Disney TV Show")
    print()
    print(document.document)


if __name__ == "__main__":
    asyncio.run(main())
