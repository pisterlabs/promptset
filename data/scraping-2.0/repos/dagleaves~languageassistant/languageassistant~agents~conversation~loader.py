"""Conversation agent LLM chain loader"""
from typing import Optional

from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

from languageassistant.agents.conversation.agent import ConversationAgent

PROMPT_TEMPLATE = (
    "Assistant is a native {language} language teacher. "
    "that teaches by using realistic conversations and  scenarios. "
    "Assistant is in a face-to-face verbal conversation with the student. "
    "Assistant should actively guide the conversation to teach the student {language} "
    "Assistant must end any replies with <END_REPLY>. "
    "If the student wishes to end the conversation or move on to a new topic, "
    "Assistant must reply with <END_CONVERSATION>. "
    "The following is a conversation between Assistant and a student. "
    "Provide Assistant's next response. "
    "\nConversation topic: {topic}\nStudent proficiency: {proficiency}\n"
    "\n{history}\nHuman: {human_input}\nAssistant:"
)


def load_conversation_agent(
    llm: BaseLanguageModel,
    prompt_template: Optional[str] = None,
    verbose: bool = False,
) -> ConversationAgent:
    """
    Return a conversation agent initialized with memory and custom prompts

    Parameters
    ----------
    llm
        Which LLM to use for inference
    prompt_template
        Prompt template instructing LLM role and response instructions
    verbose
        If the LLM chain should be verbose

    Returns
    -------
    ConversationAgent
        ConversationAgent instance
    """
    if prompt_template is None:
        prompt_template = PROMPT_TEMPLATE
    prompt = PromptTemplate(
        input_variables=["language", "topic", "proficiency", "history", "human_input"],
        template=prompt_template,
    )
    memory = ConversationBufferMemory(
        memory_key="history",
        input_key="human_input",
        ai_prefix="Assistant",
        human_prefix="Human",
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=verbose)
    return ConversationAgent(llm_chain=llm_chain, stop=["<END_REPLY>"])
