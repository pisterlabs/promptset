from abc import ABC
from typing import Dict, Any, Optional

import requests
from langchain import BasePromptTemplate, PromptTemplate
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import PromptValue
from langchain.schema.runnable import RunnableConfig

from xpuls.client.models import XPPrompt


class XPChatPromptTemplate(ChatPromptTemplate, ABC):
    @classmethod
    def from_template(cls, xp_prompt: XPPrompt, **kwargs: Any) -> ChatPromptTemplate:

        """
        Overloaded method to create a custom chat prompt template from a template string.

        Args:
            xp_prompt: An XPPrompt instance.
            **kwargs: Additional keyword arguments.

        Returns:
            An instance of CustomChatPromptTemplate.
        """
        # Perform custom actions or modifications here
        # For example, you might want to handle xp_prompt or kwargs differently

        kwargs["prompt_id"] = xp_prompt.prompt_id
        kwargs["prompt_external_id"] = xp_prompt.prompt_external_id
        kwargs["prompt_version_id"] = xp_prompt.prompt_version_id
        instance = super().from_template(xp_prompt.prompt, **kwargs)
        return instance
