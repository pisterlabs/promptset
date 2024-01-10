from dataclasses import dataclass, asdict
from typing import Iterable

import openai
from openai.openai_object import OpenAIObject

from goldpot import DefaultConfig
from goldpot.CompletionEnginesConfig import CompletionEngineConfig
from goldpot.Model import CompletionInput, CompletionOutput, SearchInput, SearchOutput


@dataclass
class Goldpot:
    api_key: str
    completion_engine_config: CompletionEngineConfig = DefaultConfig.completion_engine_config
    search_engine_config: str = DefaultConfig.search_engine_config

    def __post_init__(self):
        openai.api_key = self.api_key

    def complete_(self, prompt: str) -> OpenAIObject:
        return openai.Completion.create(prompt=prompt, **asdict(self.completion_engine_config))

    def complete(self, input: CompletionInput) -> CompletionOutput:
        prompts = input.get_prompts()
        return CompletionOutput([self.complete_(prompt) for prompt in prompts], prompts)

    def search_(self, query: str, docs: Iterable[str]) -> OpenAIObject:
        return openai.Engine(str(self.search_engine_config)).search(documents=docs, query=query)

    def search(self, input: SearchInput) -> SearchOutput:
        return SearchOutput(
            #[self.search_(q := query_docs[0], ds := query_docs[1]) for query_docs in input.get_docs()], input.docs)
            # Python 3.7:
            [self.search_(query_docs[0], query_docs[1]) for query_docs in input.get_docs()], input.docs)
