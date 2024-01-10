import json
import logging
from typing import List, NamedTuple, Tuple

import numpy as np
from constants import SUMMARY_LOGGER_NAME
from openai_client import chat_api, embed_api

summary_logger = logging.getLogger(SUMMARY_LOGGER_NAME)


class Article(NamedTuple):
    title: str
    text: str
    link: str


class Story:

    def __init__(self, source: Article, title="", summary="") -> None:
        self.sources: List[Article] = [source]
        self.summary: str = summary
        self.title: str = title
        self.id: str = ""
        if not summary:
            self.update_summary()

    def add_source(self, source: Article) -> None:
        self.sources.append(source)
        self.update_summary()

    def build_prompt(self) -> str:
        compiled_sources = "\n".join([
            f"======Source {i}======:\n\n{source.text}"
            for i, source in enumerate(self.sources)
        ])
        prompt = f"""Summarize the following source(s) into a single story. The summary should be a consolidation of just the raw facts contained in the sources. Ideally, the summary is concise and easy to digest. 
        Try to write the summary using bullet points which can be rendered in markdown.
        Return a title and summary in json format as follows:
        {{
            "title": "title of the story",
            "summary": "summary of the story"
        }}

        Here are the sources:

        {compiled_sources}

        Output:
        """
        return prompt

    def update_summary(self):
        prompt = self.build_prompt()
        output = {}
        try:
            output = json.loads(chat_api(prompt))
        except json.decoder.JSONDecodeError:
            summary_logger.error("Error decoding JSON\n")
            return
        self.summary = output["summary"]
        self.title = output["title"]
        summary_logger.info("Updated summary for story (%s)\n%s", self.title,
                            self.summary)

    @classmethod
    def from_tuple(cls, title: str, body: str, citations: List[Tuple[str, str,
                                                                     str]]):
        story = cls(Article("", "", ""), title, body)
        story.sources = [
            Article(citation["title"], citation["body"], citation["url"])
            for citation in citations
        ]
        return story
