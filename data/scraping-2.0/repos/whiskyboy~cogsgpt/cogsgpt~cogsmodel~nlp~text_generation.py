from __future__ import annotations

from typing import List
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import BaseMessage

from cogsgpt.llm import LLMManager
from cogsgpt.cogsmodel import BaseModel


class TextGenerationModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self._task_name = "text-generation"

        self._llm = LLMManager().LLM
        self._prompt = self._create_prompt()

    def _create_prompt(self) -> ChatPromptTemplate:
        human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="You are a {{ task_name }} system, the arguments are {{ task_args }}. Just help me do {{ task_name }} and give me the result. The result should focus only on the {{ task_name}} task, no additional notes, and must be in text form without any urls.",
                input_variables=["task_name", "task_args"],
                template_format="jinja2",
            )
        )

        return ChatPromptTemplate.from_messages([human_message_prompt])

    def _format_prompt(self, **kwargs) -> List[BaseMessage]:
        return self._prompt.format_prompt(
            task_name=self._task_name,
            task_args=kwargs
        ).to_messages()

    def run(self, *args, **kwargs) -> str:
        request = self._format_prompt(**kwargs)
        return self._llm(request).content
    

class GenerativeTextSummarizationModel(TextGenerationModel):
    def __init__(self) -> None:
        super().__init__()
        self._task_name = "text-summarization"


class GenerativeTextTranslationModel(TextGenerationModel):
    def __init__(self) -> None:
        super().__init__()
        self._task_name = "text-translation"