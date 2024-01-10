from typing import TypedDict

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel

from little_turtle.chains import ChainAnalytics
from little_turtle.prompts import STORY_SUMMARIZATION_PROMPT_TEMPLATE
from little_turtle.services import AppConfig


class StorySummarizationChainVariables(TypedDict):
    story: str


class StorySummarizationChain:
    def __init__(self, llm: BaseLanguageModel, chain_analytics: ChainAnalytics, config: AppConfig):
        self.chain_analytics = chain_analytics
        self.llm_chain = LLMChain(
            prompt=PromptTemplate.from_template(STORY_SUMMARIZATION_PROMPT_TEMPLATE),
            llm=llm,
            verbose=config.DEBUG,
            output_key="story_event_summary",
        )

    def get_chain(self) -> LLMChain:
        return self.llm_chain

    def run(self, variables: StorySummarizationChainVariables) -> str:
        return self.llm_chain.run(variables, callbacks=[self.chain_analytics.get_callback_handler])

    @staticmethod
    def enrich_run_variables(content: str) -> StorySummarizationChainVariables:
        return StorySummarizationChainVariables(
            story=content,
        )
