from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.chains import LLMChain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from PromptKernel.IntentionRouter.ControllerAgents.Api.ApiExecutorOrchestrator import ApiExecutorOrchestrator
from PromptKernel.IntentionRouter.IntentionRoutersConstants import INTENTION_ROUTER_PROMPT
from PromptKernel.LLMClient.ChatOpenAIClient import ChatOpenAIClient
from PromptKernel.LLMClient.LLMClientConstants import GPT4MODEL
from PromptKernel.Types.UserPrompts import UserPrompts


class IntentionRouter:
    llm: BaseLanguageModel

    def __init__(self, model: str = GPT4MODEL):
        self.llm = ChatOpenAIClient(model=model).model

    def route(self, user_prompts: UserPrompts) -> str:
        llm = self.llm
        tools = [
            ApiExecutorOrchestrator(),
        ]
        prompt = PromptTemplate(
            template=INTENTION_ROUTER_PROMPT,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tool_names": ", ".join([tool.name for tool in tools]),
                "tool_descriptions": "\n".join(
                    [f"{tool.name}: {tool.description}" for tool in tools]
                ),
            },
        )

        agent = ZeroShotAgent(
            llm_chain=LLMChain(llm=llm, prompt=prompt),
            allowed_tools=[tool.name for tool in tools],
        )
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
        )

        return agent_executor.run(
            user_prompts.prompt,
            # callbacks=[total_cb],
        )
