from langchain_community.agent_toolkits.openapi.planner import RequestsGetToolWithParsing, RequestsPostToolWithParsing
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import RequestsWrapper
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from typing import Optional, List
from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.chains.llm import LLMChain

from PromptKernel.IntentionRouter.ControllerAgents.Api.Web.ApiWebConstants import API_KEY, PARSING_GET_PROMPT, PARSING_POST_PROMPT, \
    API_CONTROLLER_PROMPT
from PromptKernel.LLMClient.ChatOpenAIClient import ChatOpenAIClient
from PromptKernel.LLMClient.LLMClientConstants import GPT4MODEL


class ApiController(BaseTool):
    name = "api_controller"
    description = "Can be used to execute a plan of API calls, like api_controller(plan)"

    def _run(self, plan_str: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        loader = WebBaseLoader('https://developer.themoviedb.org/reference/search-movie')
        api_docs = loader.load()[0].page_content
        llm = ChatOpenAIClient(model=GPT4MODEL).model

        header = {
            "Authorization": API_KEY,
            "accept": "application/json",
        }
        requests_wrapper = RequestsWrapper(headers=header)
        get_llm_chain = LLMChain(llm=llm, prompt=PARSING_GET_PROMPT)
        post_llm_chain = LLMChain(llm=llm, prompt=PARSING_POST_PROMPT)
        tools: List[BaseTool] = [
            RequestsGetToolWithParsing(
                requests_wrapper=requests_wrapper, llm_chain=get_llm_chain
            ),
            RequestsPostToolWithParsing(
                requests_wrapper=requests_wrapper, llm_chain=post_llm_chain
            ),
        ]
        prompt = PromptTemplate(
            template=API_CONTROLLER_PROMPT,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "api_docs": api_docs,
                "tool_names": ", ".join([tool.name for tool in tools]),
                "tool_descriptions": "\n".join(
                    [f"{tool.name}: {tool.description}" for tool in tools]
                ),
            },
        )
        agent = ZeroShotAgent(
            llm_chain=LLMChain(llm=llm, prompt=prompt),
            allowed_tools=[tool_inside.name for tool_inside in tools],
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
        )
        return agent_executor.run(plan_str)

    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
