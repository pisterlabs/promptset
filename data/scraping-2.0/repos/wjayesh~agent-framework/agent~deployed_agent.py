from __future__ import annotations
from typing import Any, Dict

from langchain.agents import AgentExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.agent import Agent
from tools.versioned_vector_store import VersionedVectorStoreTool


class DeployedAgent(AgentExecutor):
    version: int
    # TODO make a class out of it
    deployment_config: Dict[str, Any]
    validator: Any
    memory: Any
    agent: Agent

    PROMPT: str = (
        "{prefix}"
        "Format instructions and such. tools: {tools} "
        "you should only answer in the following format: "
        "action: tool to use ..."
        "If you don't know an answer, use the action "
        "'unknwown' and empty inputs."
        "If no version is specified by the human and multiple "
        "tools for the same project are available, use the "
        "tool with the latest version always."
    )

    def get_prompt(self) -> str:
        """Returns the prompt to be used by the agent."""
        tools = self.agent.get_allowed_tools(version=self.version)
        # construct a string with all the tools and their
        # names, description and version (if the tool is of type VersionedVectorStoreTool)
        tools_str = ""
        for tool in tools:
            tools_str += f"\n{tool.name}: {tool.description} "
            if isinstance(tool, VersionedVectorStoreTool):
                tools_str += f"Version: {tool.version} \n"

        return self.PROMPT.format(prefix=self.agent.PREFIX, tools=tools)

    def __init__(
        self, agent: Agent, memory, validator, deployment_config, version
    ) -> None:
        """Initializes the agent.

        Args:
            agent: The agent to use.
            memory: The memory to use.
            validator: The validator to use.
            deployment_config: The deployment config to use.
            version: The version of the agent to use.
        """
        from langchain.chains import LLMChain
        from agent.agent import Agent
        self.version = version
        # TODO langsucks right now we're overwriting the llm chain
        # which was defined at agent definition. We should define the chain
        # once and here, ideally.
        agent.llm_chain = LLMChain(
            llm=agent.llm,
            prompt=Agent.create_prompt(agent.get_allowed_tools(version=version)),
        )
        return self.from_agent_and_tools(
            agent=agent, tools=agent.get_allowed_tools(version=version)
        )
