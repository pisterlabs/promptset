from pydantic import BaseModel, Field
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI

from langchain import LLMMathChain

llm = ChatOpenAI(temperature=0)
llm_math_chain = LLMMathChain(llm=llm, verbose=True)


class CalculatorInput(BaseModel):
    question: str = Field()


def get_calculator_tool():
    return Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        args_schema=CalculatorInput
        # coroutine= ... <- you can specify an async method if desired as well
    )
