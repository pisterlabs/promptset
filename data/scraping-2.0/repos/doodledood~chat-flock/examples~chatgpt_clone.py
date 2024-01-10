import typer
from dotenv import load_dotenv
from halo import Halo

from chatflock.backing_stores.in_memory import InMemoryChatDataBackingStore
from chatflock.base import Chat
from chatflock.conductors.round_robin import RoundRobinChatConductor
from chatflock.participants.langchain import LangChainBasedAIChatParticipant
from chatflock.participants.user import UserChatParticipant
from chatflock.renderers.terminal import TerminalChatRenderer
from examples.common import create_chat_model


def chatgpt_clone(model: str = "gpt-4-1106-preview", temperature: float = 0.0) -> None:
    chat_model = create_chat_model(model=model, temperature=temperature)

    spinner = Halo(spinner="dots")
    ai = LangChainBasedAIChatParticipant(name="Assistant", chat_model=chat_model, spinner=spinner)
    user = UserChatParticipant(name="User")
    participants = [user, ai]

    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(), renderer=TerminalChatRenderer(), initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()
    chat_conductor.initiate_dialog(chat=chat)


if __name__ == "__main__":
    load_dotenv()

    typer.run(chatgpt_clone)
