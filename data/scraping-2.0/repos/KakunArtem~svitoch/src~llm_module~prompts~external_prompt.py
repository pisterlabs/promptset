from typing import List, Union

from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import BaseMessagePromptTemplate
from langchain.schema import BaseMessage, PromptValue


class ExternalChatPromptTemplate(ChatPromptTemplate):
    """
    This class makes sure we reload external prompts to sync them with upsource before formatting prompt.
    We need it since by default python will cache values and they will not get updated after change in upsource.
    """

    messages: List[Union[BaseMessagePromptTemplate, BaseMessage]]

    def format_prompt(self, **kwargs) -> PromptValue:
        for i, message in enumerate(self.messages):
            if getattr(
                message, "external", False
            ):  # if message has external source
                self.messages[i] = message.reload()
        return super().format_prompt(**kwargs)
