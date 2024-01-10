from langchain.agents import Tool
from langchain.chains import LLMMathChain, PALChain
from langchain.tools import DuckDuckGoSearchRun


class DefaultTools:
    def __init__(self, llm):
        self.llm = llm.model
        self.palmath_tool = self._get_pal_math()
        self.llmmath_tool = self._get_llm_math()
        self.web_search = self._get_ddg_search()

    def _get_pal_math(self):
        return Tool(
            name="PAL-MATH",
            description="A language model that is really good at solving complex word math problems. Input should be a fully worded hard word math problem.",
            func=PALChain.from_math_prompt(self.llm).run,
        )

    def _get_llm_math(self):
        return Tool(
            name="Calculator",
            description="Useful for when you need to answer questions about math.",
            func=LLMMathChain.from_llm(llm=self.llm).run,
        )

    def _get_ddg_search(self):
        return Tool(
            name="Web Search",
            description="A search engine. Useful for when you need to answer questions about current events from internet. Input should be a search query.",
            func=DuckDuckGoSearchRun().run,
        )

    def as_tool(self):
        return [self.palmath_tool, self.llmmath_tool, self.web_search]
