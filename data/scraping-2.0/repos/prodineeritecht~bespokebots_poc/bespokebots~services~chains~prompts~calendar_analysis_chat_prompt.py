from typing import List, Optional
import json

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from bespokebots.services.chains.calendar_templates import (
    CalendarDataAnalyzerTemplates as Templates
)


class CalendarAnalysisChatPrompt:
    """Prompt for the CalendarAnalysisCustomChain."""

    @classmethod
    def from_user_request(cls, user_request: str) -> PromptTemplate:
        """Return a prompt from a human message."""
        
        return ChatPromptTemplate.from_messages(
            [SystemMessagePromptTemplate.from_template(Templates.AI_SYSTEM_MESSAGE),
            HumanMessagePromptTemplate.from_template(user_request)]
        )
