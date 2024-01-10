import typer
from dotenv import load_dotenv
from halo import Halo
from langchain.memory import ConversationSummaryBufferMemory

from chatflock.backing_stores import InMemoryChatDataBackingStore
from chatflock.backing_stores.langchain import LangChainMemoryBasedChatDataBackingStore
from chatflock.base import Chat, ChatDataBackingStore
from chatflock.conductors import RoundRobinChatConductor
from chatflock.participants.langchain import LangChainBasedAIChatParticipant
from chatflock.participants.user import UserChatParticipant
from chatflock.renderers import TerminalChatRenderer
from examples.common import create_chat_model, get_max_context_size


def chatgpt_clone_with_langchain_memory(model: str = "gpt-4-1106-preview", temperature: float = 0.0) -> None:
    chat_model = create_chat_model(model=model, temperature=temperature)

    spinner = Halo(spinner="dots")
    ai = LangChainBasedAIChatParticipant(name="Assistant", chat_model=chat_model, spinner=spinner)
    user = UserChatParticipant(name="User")
    participants = [user, ai]

    max_context_size = get_max_context_size(chat_model)
    if max_context_size is None:
        backing_store: ChatDataBackingStore = InMemoryChatDataBackingStore()
    else:
        memory = ConversationSummaryBufferMemory(llm=chat_model, max_token_limit=max_context_size)
        backing_store = LangChainMemoryBasedChatDataBackingStore(memory=memory)

    chat = Chat(backing_store=backing_store, renderer=TerminalChatRenderer(), initial_participants=participants)

    chat_conductor = RoundRobinChatConductor()
    chat_conductor.initiate_dialog(chat=chat)


if __name__ == "__main__":
    load_dotenv()

    typer.run(chatgpt_clone_with_langchain_memory)
