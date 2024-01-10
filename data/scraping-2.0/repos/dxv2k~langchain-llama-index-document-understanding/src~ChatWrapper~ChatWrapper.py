from threading import Lock
from typing import Optional, Tuple

from langchain.chains import ConversationChain
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory


class ChatWrapper:

    def __init__(self, 
                 agent: AgentExecutor = None):
        self.lock = Lock()
        self.agent = agent    

    def __call__(
            self, 
            inp: str, 
            history: Optional[Tuple[str, str]], 
            agent: Optional[AgentExecutor]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []

            if agent: self.agent = agent

            # If chain is None, that is because no API key was provided.
            if self.agent is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history

            # Run chain and append input.
            output = self.agent.run(input=inp)
            history.append((inp, output))

        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

    def clear_agent_memory(self) -> None: 
        if not self.agent: 
            raise ValueError("ChatWrapper doesn't have an agent yet") 

        try: 
            self.agent.memory.clear()
            memory = ConversationBufferMemory(memory_key="chat_history")
            self.agent.memory = None
            self.agent.memory = memory
        except Exception as e: 
            raise(e)