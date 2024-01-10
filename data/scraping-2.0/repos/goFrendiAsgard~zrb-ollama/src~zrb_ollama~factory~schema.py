from zrb.helper.typing import Callable
from langchain.chains import LLMChain
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.memory.chat_memory import BaseChatMemory
from langchain.callbacks.manager import CallbackManager
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import Agent, AgentExecutor, Tool
from ..task.any_prompt_task import AnyPromptTask

LLMChainFactory = Callable[[AnyPromptTask], LLMChain]
CallbackManagerFactory = Callable[[AnyPromptTask], CallbackManager]
ChatModelFactory = Callable[[AnyPromptTask], BaseChatModel]
ChatPromptTemplateFactory = Callable[[AnyPromptTask], ChatPromptTemplate]
ChatMemoryFactory = Callable[[AnyPromptTask], BaseChatMemory]
AgentToolFactory = Callable[[AnyPromptTask], Tool]
PromptTemplateFactory = Callable[[AnyPromptTask], PromptTemplate]
AgentFactory = Callable[[AnyPromptTask], Agent]
AgentExecutorFactory = Callable[[AnyPromptTask], AgentExecutor]
