from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory import (
    ConversationBufferMemory, ConversationBufferWindowMemory,
    ConversationSummaryMemory
)
from langchain_core.language_models import BaseLanguageModel
from .schema import ChatMemoryFactory
from ..task.any_prompt_task import AnyPromptTask


def chat_conversation_buffer_memory_factory() -> ChatMemoryFactory:
    def create_chat_memory(task: AnyPromptTask) -> BaseChatMemory:
        return ConversationBufferMemory(
            memory_key='chat_history', return_messages=True
        )
    return create_chat_memory


def chat_conversation_buffer_window_memory_factory(
    k: int | str = 3
) -> ChatMemoryFactory:
    def create_chat_memory(task: AnyPromptTask) -> BaseChatMemory:
        rendered_k = task.render_int(k)
        return ConversationBufferWindowMemory(
            k=rendered_k, memory_key='chat_history', return_messages=True
        )
    return create_chat_memory


def chat_conversation_summary_memory_factory(
    llm: BaseLanguageModel
) -> ChatMemoryFactory:
    def create_chat_memory(task: AnyPromptTask) -> BaseChatMemory:
        return ConversationSummaryMemory(
            llm=llm, memory_key='chat_history', return_messages=True
        )
    return create_chat_memory

