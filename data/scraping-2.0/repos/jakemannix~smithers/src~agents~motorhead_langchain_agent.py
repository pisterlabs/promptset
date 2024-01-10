import sys
import asyncio
from typing import Any, Union, Dict, List, Optional
from uuid import UUID

from langchain.callbacks.manager import CallbackManager, BaseCallbackManager, BaseCallbackHandler
from langchain.chains import ConversationChain, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from dotenv import load_dotenv
from langchain.memory.motorhead_memory import MotorheadMemory
from langchain.schema import AgentFinish, AgentAction, LLMResult, BaseMessage
from termcolor import colored

load_dotenv()


class CustomCallbackManager(BaseCallbackManager):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID,
                            parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None,
                            **kwargs: Any) -> Any:
        pass

    def add_handler(self, handler: BaseCallbackHandler, inherit: bool = True) -> None:
        pass

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        pass

    def set_handlers(self, handlers: List[BaseCallbackHandler], inherit: bool = True) -> None:
        pass

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        return sys.stdout.write(colored(token, "green"))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        pass

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        pass

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        pass

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        pass

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        pass

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        pass

    def on_text(self, text: str, **kwargs: Any) -> Any:
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        pass


async def run():
    cb_manager = CustomCallbackManager(handlers=[])  # CallbackManager(handlers=[CustomCallbackManager()])
    chat = ChatOpenAI(
        temperature=0,
        streaming=True,
        callback_manager=cb_manager
    )

    memory = MotorheadMemory(
        return_messages=True,
        memory_key="history",
        session_id="davemustaine666",
        url="http://localhost:8080",
    )
    await memory.init()

    context = ""
    if memory.context:
        context = f"\nHere's previous context: {memory.context}"

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                f"The following is a friendly conversation between a human and an AI. The AI is talkative and "
                f"provides lots of specific details from its context. If the AI does not know the answer to a "
                f"question, it truthfully says it does not know. {context}"
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    search = GoogleSearchAPIWrapper()
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events",
        )
    ]

    prefix = """
        Have a conversation with a human, answering the following questions as best you can.
        You have access to the following tools:
    """
    suffix = """Begin!"
    
    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    llm_chain = LLMChain(llm=chat, prompt=prompt, memory=memory, callback_manager=cb_manager)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )
    # chain = ConversationChain(memory=memory, prompt=chat_prompt, llm=chat)

    def post_to_bash():
        while True:
            answer_i = input(colored("", "green"))
            if not answer_i:
                continue
            response_i = chain.run(answer_i)
            print(colored(response_i, "green"))

    print(colored("\nMotorhead 🤘chat start\n", "blue"))
    answer = input(colored("", "green"))
    response = chain.run(answer)
    print(colored(response, "green"))
    post_to_bash()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt as kie:
        print(colored("\nI see you have chosen to end the conversation with me 💔. Good bye!", "yellow"))