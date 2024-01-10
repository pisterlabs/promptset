import openai
import os

from models.models import Trend
from trend.descriptor.base_descriptor import BaseTrendDescriptor

openai.api_key = os.getenv("OPENAI_API_KEY", None)
openai.base_url = os.getenv("OPENAI_API_BASE", "http://localhost:8002/v1")

model = os.getenv("OPENAI_MODEL", "gpt4")


def get_gpt_descriptor():
    return GPTDescriptor()


class GPTDescriptor(BaseTrendDescriptor):
    def generate_description(self, topics: list[str], start_year: int,
                             end_year: int, values: list[int],
                             global_trend: Trend, sub_trends: list[Trend]) -> str:

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """You are a data analyst. You are given a list of trends. Each trend represents a segment of a time series and comes with a start year, end year, trend type (no trend = 0, up = 1, down = 2) and a slope. Please shortly describe and summarize the history of the corresponding time series using this list of trends. The first trend is the global trend."""
                },
                {
                    "role": "user",
                    "content": """trends={}""".format(topics, range(start_year, end_year + 1), values, [global_trend] + sub_trends)
                }
            ],
            max_tokens=1024,
            top_p=1,
            frequency_penalty=1,
            presence_penalty=0
        )

        return response.choices[0].message.content
