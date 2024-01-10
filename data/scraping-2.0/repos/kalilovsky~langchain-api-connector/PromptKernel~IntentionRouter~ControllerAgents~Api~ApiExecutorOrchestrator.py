from typing import Optional

from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.chains import LLMChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from PromptKernel.IntentionRouter.ControllerAgents.Api.ApiExecutorOrchestratorConstants import API_ORCHESTRATOR_PROMPT
from PromptKernel.IntentionRouter.ControllerAgents.Api.Web.ApiController import ApiController
from PromptKernel.IntentionRouter.ControllerAgents.Api.Web.ApiDecisionPlanner import ApiDecisionPlanner
from PromptKernel.LLMClient.ChatOpenAIClient import ChatOpenAIClient
from PromptKernel.LLMClient.LLMClientConstants import GPT4MODEL


class ApiExecutorOrchestrator(BaseTool):
    name = "api_executor_orchestrator"
    description = "Can be used to execute a plan to execute post and get api calls, like api_executor_orchestrator(plan)."

    def _run(
            self, prompt: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        llm = ChatOpenAIClient(model=GPT4MODEL).model
        query = prompt

        tools = [
            ApiDecisionPlanner(),
            ApiController(),
        ]
        prompt_template = PromptTemplate(
            template=API_ORCHESTRATOR_PROMPT,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tool_names": ", ".join([tool.name for tool in tools]),
                "tool_descriptions": "\n".join(
                    [f"{tool.name}: {tool.description}" for tool in tools]
                ),
            },
        )
        agent = ZeroShotAgent(
            llm_chain=LLMChain(llm=llm, prompt=prompt_template),
            allowed_tools=[tool.name for tool in tools],
        )
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
        )

        return agent_executor.run(
            query,
            # callbacks=[total_cb],
        )

    async def _arun(
            self, api_spec: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("api_executor_orchestrator does not support async")
