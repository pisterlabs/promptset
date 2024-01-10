import typer
from dotenv import load_dotenv
from halo import Halo

from chatflock.backing_stores import InMemoryChatDataBackingStore
from chatflock.base import Chat
from chatflock.conductors import LangChainBasedAIChatConductor, RoundRobinChatConductor
from chatflock.participants.group import GroupBasedChatParticipant
from chatflock.participants.langchain import LangChainBasedAIChatParticipant
from chatflock.participants.user import UserChatParticipant
from chatflock.renderers import TerminalChatRenderer
from examples.common import create_chat_model


def manual_hierarchical_participant(model: str = "gpt-4-1106-preview", temperature: float = 0.0) -> None:
    chat_model = create_chat_model(model=model, temperature=temperature)
    spinner = Halo(spinner="dots")
    comedy_team = GroupBasedChatParticipant(
        group_name="Comedy Team",
        mission="Collaborate on funny humour-filled responses based on the original request for the user",
        chat=Chat(
            backing_store=InMemoryChatDataBackingStore(),
            renderer=TerminalChatRenderer(),
            initial_participants=[
                LangChainBasedAIChatParticipant(
                    name="Bob",
                    role="Chief Comedian",
                    personal_mission="Take questions from the user and collaborate with "
                    "Tom to come up with a succinct funny (yet realistic) "
                    "response. Short responses are preferred.",
                    chat_model=chat_model,
                    spinner=spinner,
                ),
                LangChainBasedAIChatParticipant(
                    name="Tom",
                    role="Junior Comedian",
                    personal_mission="Collaborate with Bob to come up with a succinct "
                    "funny (yet realistic) response to the user. Short responses are preferred",
                    chat_model=chat_model,
                    spinner=spinner,
                ),
            ],
        ),
        chat_conductor=LangChainBasedAIChatConductor(
            chat_model=chat_model,
            goal="Come up with a funny succinct response to the user.",
            interaction_schema="Bob should collaborate with Tom to come up with a great funny and succinct response. "
            "When one is agreed upon, the chat should end",
            spinner=spinner,
        ),
        spinner=spinner,
    )
    user = UserChatParticipant(name="User")
    participants = [user, comedy_team]
    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(), renderer=TerminalChatRenderer(), initial_participants=participants
    )
    chat_conductor = RoundRobinChatConductor()
    chat_conductor.initiate_dialog(chat=chat)


if __name__ == "__main__":
    load_dotenv()
    typer.run(manual_hierarchical_participant)
