import instructor
import datetime
import enum
import asyncio
from typing import List
from openai import OpenAI
from pydantic import BaseModel


class ClientSource(enum.Enum):
    GMAIL = "gmail"
    CALENDAR = "calendar"

class SearchClient(BaseModel):
    query: str
    keywords: List[str]
    email: str
    source: ClientSource
    start_date: datetime.date
    end_date: datetime.date

    async def execute(self) -> str:
        if self.source == ClientSource.GMAIL:
            return "hello"
        elif self.source == ClientSource.CALENDAR:
            return "bye"

class Retrival(BaseModel):
    queries: List[SearchClient]

    async def execute(self) -> str:
        return await asyncio.gather(*[query.execute() for query in self.queries])

# Enables response_model in the openai client
client = instructor.patch(OpenAI())

retrival = client.chat.completions.create(
    model="gpt-4",
    response_model=Retrival,
    messages=[
        {"role": "system", "content": "You are Jason's personal assistant."},
        {"role": "user", "content": "What do I have today?"}
    ],
)

print(retrival)