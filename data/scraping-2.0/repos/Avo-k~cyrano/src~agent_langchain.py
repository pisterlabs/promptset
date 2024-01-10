import os
from datetime import datetime
from queue import Empty, Queue
from threading import Thread
from typing import Any, Generator, Union

from dotenv import load_dotenv
from langchain import GoogleSerperAPIWrapper
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import (
    CombinedMemory,
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
)
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.tools.python.tool import PythonREPLTool

from memories.long_term_memory import init_memory
from tools._chess import GetBoardState, PlayMove, ResetBoard
from tools._timer import TimerTool

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]

# Create a Queue
Q = Queue()


# Defined a QueueCallback, which takes as a Queue object during initialization. Each new token is pushed to the queue.
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: Any) -> None:
        return self.q.empty()


llm = ChatOpenAI(
    temperature=0.3,
    # model="gpt-4-0613",
    model="gpt-3.5-turbo-0613",
    streaming=True,
    callbacks=[QueueCallback(Q)],
)
search = GoogleSerperAPIWrapper()

tools = [
    Tool(
        name="RechercheGoogle",
        func=search.run,
        description="utile lorsque tu dois répondre à des questions sur des événements d'actualité. pose des questions ciblées.",
    ),
    PythonREPLTool(),
    # Timer
    TimerTool(),
    # Chess
    PlayMove(),
    GetBoardState(),
    ResetBoard(),
]

memory = init_memory(max_token_limit=2000)

current_date = datetime.today().strftime("%d/%m/%Y")
system_message = SystemMessage(content=os.getenv("SYS_PROMPT").format(current_date=current_date))

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}


# Create a function that will return our generator
def stream(agent, input_text) -> Generator:
    with Q.mutex:
        Q.queue.clear()
    job_done = object()

    # Create a function to call - this will run in a thread
    def task():
        resp = agent.run(input_text)
        Q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = Q.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            yield next_token, content
        except Empty:
            continue


def stream_sentences(agent, input_text) -> Generator:
    """wrapper to stream function"""
    sentence = ""
    for next_token, content in stream(agent, input_text):
        sentence += next_token
        if "\n\n" in next_token:
            yield sentence
            sentence = ""
    if sentence:
        yield sentence


def init_agent():
    return initialize_agent(
        tools=tools,
        llm=llm,
        streaming=True,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        max_iterations=5,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )
