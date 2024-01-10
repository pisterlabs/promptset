#
# WIP and untested
#


import langchain.chat_models
import numpy as np
import openai
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from ....context.data import FormatStr
from ....lang_models import count_tokens, query_model
from ...event import Event
from ..base import MemoryBase


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class RelevanceTextMemoryItem:
    event: Event
    text: str
    embedding: np.ndarray
    token_count: int
    summary_text: str
    summary_token_count: int


class RelevanceTextMemory(MemoryBase):
    def __init__(
        self,
        embed_model=None,
        text_model=None,
    ):
        self.embed_model = embed_model or openai.Embedding()
        self.text_model = text_model or langchain.chat_models.ChatOpenAI()
        self.summary_limit = 300
        self.summary_request = "100 words"
        self.events = []

    def add_event(self, event: Event):
        # TODO: Consider less naive text to embed - e.g. formatted
        text = str(event)
        tc = count_tokens(text)
        emb = self.embed_model.embed_documents([text])[0]
        emb = emb / np.sum(emb**2) ** 0.5  # Normalize to L_2(emb)==1
        if tc > self.summary_limit:
            summary = query_model(
                self.text_model,
                FormatStr("Summarize the following text {length}:\n---\n{text}").format(
                    length=self.summary_request, text=text
                ),
            )
        self.events.append(
            RelevanceTextMemoryItem(
                event, text=text, embedding=emb, token_count=token_count
            )
        )

    def get_events(
        self, query: str = None, max_events: int = 10, max_tokens: int = None
    ) -> tuple[Event]:
        assert max_tokens is None
        emb = self.embed_model.embed_query(str(query))
        emb = emb / np.sum(emb**2) ** 0.5
        scores = [np.dot(emb, e.embedding) for e in self.events]
        cutoff = np.sort(scores)[-max_events]
        return tuple(e for e, s in zip(self.events, scores) if s >= cutoff)
