from __future__ import annotations
from rich import print
from uuid import UUID
from typing import Any, List
from langchain import LLMChain
from langchain.agents import Tool
from langchain.chat_models import AzureChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish


suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""


class DialogueHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.memory: List[List[str]] = [[]]

    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.memory.append([])
        return super().on_agent_finish(finish, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_agent_action(self, action: AgentAction, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        self.memory[-1].append(action.log)
        return super().on_agent_action(action, run_id=run_id, parent_run_id=parent_run_id, **kwargs)


class DialogWrapper:
    def __init__(
            self, llm: AzureChatOpenAI, toolModels: List,
            customedPrefix: str, verbose: bool = False
    ) -> Any:
        self.d_handler = DialogueHandler()
        tools = []

        for ins in toolModels:
            func = getattr(ins, 'embody')
            tools.append(
                Tool(
                    name=func.name,
                    description=func.description,
                    func=func
                )
            )

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=customedPrefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )
        self.agent_memory = ConversationBufferMemory(memory_key="chat_history")

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        agent = ZeroShotAgent(
            llm_chain=llm_chain,
            tools=tools, verbose=verbose
        )
        self.agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools,
            verbose=verbose, memory=self.agent_memory,
            handle_parsing_errors="Use the TALM output directly as the final answer!"
        )

    def dialogue(self, input: str):
        print('TALM is thinking, one sec...')
        with get_openai_callback() as caller:
            # actually start a agent_chain to query:
            response = self.agent_chain.run(input=input, callbacks=[self.d_handler])
        # print('History: ', self.agent_memory.buffer)
        return response, caller
