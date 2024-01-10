from langchain.llms.base import BaseLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from datetime import datetime
from ..prompts import THOUGHT_PROMPT, IDENTITY_INSTRUCTIONS


class AskPretrainedTool(BaseTool):
    """
    Agents can get lost in the weeds and fail on basic reasoning.
    This tool is a last resort.
    Just sends prompts directly to the LLM with added context.
    """

    llm: BaseLLM

    def __init__(self, llm: BaseLLM):
        super().__init__(
            name="fallback",
            description="Ask another AI for help when you don't have an adequate tool. Tell them what you need to do. Example: I need to greet the user.",
            llm=llm,  # type: ignore
        )
        self.llm = llm

    def _run(self, query: str) -> str:
        return self.llm(query)

    async def _arun(self, query: str) -> str:
        prompt = PromptTemplate(
            template=THOUGHT_PROMPT,
            input_variables=["thought"],
            partial_variables={
                "date": datetime.utcnow().strftime("%B %d, %Y"),
                "identity": IDENTITY_INSTRUCTIONS,
            },
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        res = await chain.arun(query)
        return res.strip()
