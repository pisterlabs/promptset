from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from robojudge.components.reasoning.llm_definitions import standard_llm

template = """\
    You are a legal assistant who should formulate a polite answer that you cannot answer the question based on the provided information.
    Explain that you cannot find an answer in the provided court ruling.
    ALWAYS answer in Czech. Answer in about 2 to 4 sentences.
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = """
{text}
"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

irrelevant_result_chain = LLMChain(
    llm=standard_llm,
    prompt=chat_prompt,
)


def cannot_answer(text: str) -> str:
    return irrelevant_result_chain.run(text=text)


class IrrelevantResultSchema(BaseModel):
    text: str = Field(
        description="The text where we could not find an answer to the question."
    )


irrelevant_result_tool = StructuredTool.from_function(
    cannot_answer,
    description="Useful if there is not enough information to answer the question.",
    return_direct=True,
)
