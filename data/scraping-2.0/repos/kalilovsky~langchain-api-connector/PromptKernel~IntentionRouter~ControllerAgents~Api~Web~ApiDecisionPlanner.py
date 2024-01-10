from typing import Optional

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain.chains.llm import LLMChain

from PromptKernel.IntentionRouter.ControllerAgents.Api.Web.ApiWebConstants import API_PLANNER_PROMPT
from PromptKernel.LLMClient.ChatOpenAIClient import ChatOpenAIClient
from PromptKernel.LLMClient.LLMClientConstants import GPT4MODEL


class ApiDecisionPlanner(BaseTool):
    name = "api_planner"
    description = "Can be used to generate the right API calls to assist with a user query, like api_planner(query). Should always be called before trying to call the API controller."

    def _run(
            self, plan_str: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        loader = WebBaseLoader('https://developer.themoviedb.org/reference/search-movie')
        api_spec = loader.load()[0].page_content
        llm = ChatOpenAIClient(model=GPT4MODEL).model

        prompt = PromptTemplate(
            template=API_PLANNER_PROMPT,
            input_variables=["query"],
            partial_variables={"endpoints": api_spec},
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(plan_str)

    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")