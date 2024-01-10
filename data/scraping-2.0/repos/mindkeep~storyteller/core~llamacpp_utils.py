"""main module for the storyteller application"""

from langchain.llms.base import LLM
from langchain.llms.llamacpp import LlamaCpp
from langchain.memory.chat_memory import BaseChatMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from core import config
from ui.baseui import BaseUI


PROMPT_TEMPLATE = """You are a dungeon master and responding
to the player's stated actions.

Example interaction:
AI: You enter the tavern and sees a barkeep and a few adventures whispering
to each other over drinks at a round table.
Human: I walk over to the table ask if I can join them.
AI: They look you up and down and decide to ignore you and continue their
conversation. Behind you, you hear the barkeep laugh.

Conversation history:
{memory}
Human: {response}
AI: """

def init_llamacpp(model: config.LlamaCppModel, interface: BaseUI) -> LlamaCpp:
    """ Initialize the LLM

    Args:
        conf (config.Config): the configuration

    Returns:
        LlamaCpp: the LLM
    """
    # initialize the LLM
    llm = LlamaCpp(
        model_path=model.path,
        callback_manager=interface.get_callback_manager(),
        verbose=True,
        n_ctx=model.n_ctx,
        temperature=model.temperature,
        stop=["\n\n", "You:", "you:", "Player:", "player:", "Human:", "human:"],
    )  # type: ignore
    return llm

def init_chain(
        llm: LLM,
        interface: BaseUI,
        memory: BaseChatMemory) -> LLMChain:
    """ Initialize the LLM chain

    Args:
        conf (config.Config): the configuration
        interface (BaseUI): the interface

    Returns:
        LLMChain: the LLM chain
    """

    # create our prompt template
    prompt_template = PromptTemplate(
        input_variables=["memory", "response"],
        template=PROMPT_TEMPLATE)


    # initialize the conversation chain
    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
        verbose=True,
        callback_manager=interface.get_callback_manager(),
        )

    return conversation_chain
