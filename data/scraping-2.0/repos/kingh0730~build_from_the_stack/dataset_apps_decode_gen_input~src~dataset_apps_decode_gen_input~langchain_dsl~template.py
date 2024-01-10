from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import openai
from . import (
    OPENAI_API_KEY,
    SYSTEM_TEMPLATE,
    HUMAN_TEMPLATE,
    QUESTION_2,
    DSL_2,
    QUESTION_1,
    DSL_1,
)


def build_dsl_template():
    system_message_prompt = SystemMessage(content=SYSTEM_TEMPLATE)
    human_message_prompt = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    return chat_prompt


def dsl_template(question: str):
    chat_prompt = build_dsl_template()
    msg = chat_prompt.format_messages(
        question=question,
    )
    return msg


def dsl_chain(question: str, *, model: str):
    chat = ChatOpenAI(
        model=model,
        openai_api_key=OPENAI_API_KEY,
    )

    chat_prompt = build_dsl_template()

    chain = LLMChain(
        llm=chat,
        prompt=chat_prompt,
    )
    ans = chain.run(
        question=question,
    )
    return ans
