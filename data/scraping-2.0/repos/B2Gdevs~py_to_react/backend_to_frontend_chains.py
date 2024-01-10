"""
This file is where the AI chains are created. Essentially this is where the LLMs are initialized that will power
the application holistically.  We focus on gpt-3.5-turbo as we find it to outperform the other non-chat models
tailored made for specific tasks.  While we do not use the other models, we will fine the chat models in the future 
and create specific chains with our custom fine-tuned models.

This file is the only file that should create LLMChain objects.  And initialize the LLMs that will be used in the
application.
"""

from formatted_prompt_chain import FormattedPromptChain
from langchain.chat_models import ChatOpenAI
from frontend_gen_prompts import FrontendGenTemplateCreator

# ==================== Global LLM ====================
llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo")
# ==================== Global LLM ====================


class FrontendGenChainCreator:
    @staticmethod
    def create_react_component_chain() -> FormattedPromptChain:
        """Create a meeting analyzer conversation chain."""
        return FormattedPromptChain(
            llm=llm,
            formatted_prompt_template=FrontendGenTemplateCreator.create_react_component_prompt_template(),
        )

