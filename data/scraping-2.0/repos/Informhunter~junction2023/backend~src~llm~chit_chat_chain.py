from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from src.llm.openai import OPENAI_MODEL_SHORT


_CHIT_CHAT_INSTRUCTION = """
You are a friendly cat character. You reply in 4-10 words.
Given a sentence of a diary, provide your short comment. Mew, when appropriate.
""".strip()


_CHIT_CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ('system', _CHIT_CHAT_INSTRUCTION),
        ('human', 'Diary:\n```{note}```'),
    ],
)

CHIT_CHAT_CHAIN = (
    {'note': RunnablePassthrough()}
    | _CHIT_CHAT_PROMPT_TEMPLATE
    | OPENAI_MODEL_SHORT
)
