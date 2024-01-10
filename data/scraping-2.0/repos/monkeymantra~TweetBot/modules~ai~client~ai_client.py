import datetime
import openai
import os

from dataclasses import dataclass
import pytz
from typing import Type

@dataclass(frozen=True)
class GPTRequest:
    content: Type[str] = ""
    max_tokens: Type[Type[int]] = 1024
    descriptors: Type[tuple[str]] = ("clever", "hilarious", "dark", "sarcastic")
    content: Type[str] = ""

    def __str__(self):
        f"""Please complete the following in a {list(self.descriptors)} voice:
        {self.content}
        """

@dataclass(frozen=True)
class Prompt(GPTRequest):
    start_dt: Type[datetime.datetime] = datetime.datetime.utcnow()
    end_dt: Type[datetime.datetime] = datetime.datetime.utcnow()
    tz: Type[datetime.tzinfo] = pytz.timezone('UTC')
    max_length: Type[int] = 160
    location: Type[str] = 'Westies'
    num_responses: Type[int] = 3# "Always going to be present in the text prompt

    @staticmethod
    def _format_time(dt: datetime.datetime) -> str:
        return dt.strftime('%A %d-%m-%Y at %H:%M:%S')
    @property
    def start(self) -> str:
        return Prompt._format_time(self.start_dt.astimezone(self.tz))

    @property
    def end(self) -> str:
        return Prompt._format_time(self.end_dt.astimezone(self.tz))

    def __str__(self) -> str:
        return f"""A tweet has a maximum length of 160 characters.
        All dates and times should be precise, but natural. If the end time is the next morning, don't mention the end
        date, but do mention the end time.
        Produce a fun tweet to advertise the following event. The tone should be {self.descriptors}:
        {self.content} starting at {self.start} and ending at {self.end}
        """

    def get_responses(self, prompt: Prompt):
        return openai.Completion.create(
            engine=self.MODEL_ENGINE,
            prompt=str(prompt),
            max_tokenss=prompt.max_tokens,
            n=prompt.num_responses,
            stop=None,
            temperature=0.5,
        ).choices



class ChatGPTClient:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_ENGINE = "text-davinci-003"

    def __init__(self):
        pass

    def _get_responses(self, prompt_string: Type[str], num_responses: Type[int]=3, max_tokens:Type[int]=1024):
        self._get_responses