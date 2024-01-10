from abc import abstractmethod
from zrb import AnyTask
from zrb.helper.typing import Any, List, Mapping
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.memory.chat_memory import BaseChatMemory
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import Agent, AgentExecutor, Tool
from langchain.callbacks.manager import CallbackManager


class AnyPromptTask(AnyTask):

    @abstractmethod
    def get_agent_executor(self) -> AgentExecutor | None:
        pass

    @abstractmethod
    def get_agent_llm_chain(self) -> LLMChain | None:
        pass

    @abstractmethod
    def get_agent(self) -> Agent | None:
        pass

    @abstractmethod
    def get_agent_prompt_template(self) -> PromptTemplate | None:
        pass

    @abstractmethod
    def get_agent_tools(self) -> List[Tool] | None:
        pass

    @abstractmethod
    def get_llm_chain(self) -> LLMChain | None:
        pass

    @abstractmethod
    def get_callback_manager(self) -> CallbackManager | None:
        pass

    @abstractmethod
    def get_chat_model(self) -> BaseChatModel | None:
        pass

    @abstractmethod
    def get_chat_prompt_template(self) -> ChatPromptTemplate | None:
        pass

    @abstractmethod
    def get_chat_memory(self) -> BaseChatMemory | None:
        pass

    @abstractmethod
    def load_chat_context_to_memory(
        self, memory: BaseChatMemory
    ) -> BaseChatMemory | None:
        pass

    @abstractmethod
    def save_chat_context(self, input: Any, output: Any):
        pass

    @abstractmethod
    def get_chat_context(self) -> List[Mapping[str, Mapping[str, Any]]] | None:
        pass

    @abstractmethod
    def get_rendered_history_file_name(self) -> str | None:
        pass

    @abstractmethod
    def get_rendered_user_prompt(self) -> Any:
        pass

    @abstractmethod
    def get_rendered_system_prompt(self) -> str:
        pass
