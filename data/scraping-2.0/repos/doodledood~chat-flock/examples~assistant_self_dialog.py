import typer
from dotenv import load_dotenv
from halo import Halo

from chatflock.backing_stores.in_memory import InMemoryChatDataBackingStore
from chatflock.base import Chat
from chatflock.conductors.round_robin import RoundRobinChatConductor
from chatflock.participants.langchain import LangChainBasedAIChatParticipant
from chatflock.renderers.terminal import TerminalChatRenderer
from examples.common import create_chat_model


def assistant_self_dialog(model: str = "gpt-4-1106-preview", temperature: float = 0.0) -> None:
    chat_model = create_chat_model(model=model, temperature=temperature)

    spinner = Halo(spinner="dots")
    writer = LangChainBasedAIChatParticipant(
        name="Novel Writer",
        role="Novel Writer",
        personal_mission="Write great, romantic novels. Be the best writer in the world.",
        chat_model=chat_model,
        spinner=spinner,
    )

    # There could only be one participant in this chat, in that case, it will be more like a scratchpad for the
    # participant to write down their thoughts, do self-reflection, and actual concrete work.
    participants = [writer]

    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(), renderer=TerminalChatRenderer(), initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()
    chat_conductor.initiate_dialog(
        chat=chat,
        initial_message="I want to write and complete my short 2-page long novel about space turtles now. "
        'Space turtles are AWESOME. When I am done with it I should respond with the word "TERMINATE" no quotes with '
        "nothing else after it",
    )


if __name__ == "__main__":
    load_dotenv()

    typer.run(assistant_self_dialog)
