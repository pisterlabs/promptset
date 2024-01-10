from dataclasses import dataclass
from typing import Tuple, List

import openai

from crisis import RssSource, Feed, rss_sources
from crisis.prompt import prompt


@dataclass
class GPT3Api:
    api_key: str
    engine: str
    temperature: float
    max_tokens: int

    def get_summarization(self, topic: str, question: str) -> Tuple[str, List[str]]:
        openai.api_key = self.api_key

        rss_name_to_src = {r.name: r for r in rss_sources}

        feed = Feed.from_rss_source(rss_name_to_src[topic])

        return prompt(
            feed=feed,
            question=question,
            model=self.engine,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
