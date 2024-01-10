from halo import Halo
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory

from ..llm import get_llm
from config import Models


def format_initial_prompt(prompt: str, file_content: str) -> str:
    return f"""
    Instructions: {prompt}

    {file_content}
    """


prompt = """
Respond with a single question that you would need to ask to gain more clarity about how to follow the most recent instructions or feedback. Return just the question. If everything is clear, return the string "nothing left to clarify". You have been trusted to make assumptions, not every small detail needs to be clarified.

Chat History: 
{chat_history}
"""


@Halo(text="Interpreting", spinner="dots")
def ask_for_clarification(memory: str):
    chain = LLMChain(
        llm=get_llm(Models.CONVERSATION_MODEL),
        prompt=PromptTemplate.from_template(prompt),
    )

    return chain.predict(chat_history=memory)
