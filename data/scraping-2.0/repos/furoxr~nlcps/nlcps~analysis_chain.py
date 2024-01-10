from typing import Dict, List

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic.v1 import BaseModel, PrivateAttr

from nlcps.types import AnalysisExample, AnalysisResult, ContextRuleExample

ANALYSIS_PROMPT = (
    "There are {entities_len} entities in the utterance: {entities}."
    "You need to perform the following tasks:\n"
    "1. Categorize a given sentence into entity categories. Each sentence can have more than one category.\n"
    "2. Classify whether a sentence requires context."
    "Context is required when additional information about the content of a presentation is required"
    "to fulfill the task described in the sentence\n"
    "{context_rules}\n\n"
    "{format_instructions}"
    "Let's think step by step."
)


class AnalysisChain(BaseModel):
    llm: ChatOpenAI
    entities: List[str]

    _prompt_template: ChatPromptTemplate = PrivateAttr()
    _chain: LLMChain = PrivateAttr()

    async def init_chain(self) -> None:
        """Initialize an analysis chain."""
        example_prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template("{input}"),
                AIMessagePromptTemplate.from_template("{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt, examples=await self.format_examples()
        )
        final_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(ANALYSIS_PROMPT),
                few_shot_prompt,
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        parser = PydanticOutputParser(pydantic_object=AnalysisResult)

        context_rules = await ContextRuleExample.scroll(None)
        self._prompt_template = final_prompt.partial(
            entities_len=len(self.entities),  # type: ignore
            entities=",".join(self.entities),
            context_rules="\n".join([i.rule for i in context_rules]),
            format_instructions=parser.get_format_instructions(),
        )

        self._chain = LLMChain(
            llm=self.llm,
            prompt=self._prompt_template,
            output_parser=parser,
        )

    async def format_examples(self) -> List[Dict[str, str]]:
        """Handle the format of examples"""
        examples = await AnalysisExample.scroll(None)
        return [
            {"input": e.utterance, "output": e.analysis_result.json()}  # type: ignore
            for e in examples
        ]

    async def run(self, user_utterance: str) -> AnalysisResult:
        await self.init_chain()
        return self._chain.run(user_utterance)
