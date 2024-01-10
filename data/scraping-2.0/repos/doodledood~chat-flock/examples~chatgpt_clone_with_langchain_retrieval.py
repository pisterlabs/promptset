import typer
from dotenv import load_dotenv
from halo import Halo
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from chatflock.backing_stores import InMemoryChatDataBackingStore
from chatflock.base import Chat
from chatflock.conductors import RoundRobinChatConductor
from chatflock.participants.langchain import LangChainBasedAIChatParticipant
from chatflock.participants.user import UserChatParticipant
from chatflock.renderers import TerminalChatRenderer
from examples.common import create_chat_model


def chatgpt_clone_with_langchain_retrieval(model: str = "gpt-4-1106-preview", temperature: float = 0.0) -> None:
    chat_model = create_chat_model(model=model, temperature=temperature)

    spinner = Halo(spinner="dots")

    # Set up a simple document store.
    texts = [
        "The user's name is Eric.",
        "The user likes to eat Chocolate.",
        "The user loves to play video games.",
        "The user is a software engineer.",
    ]

    # Make sure you install chromadb: `pip install chromadb`
    db = Chroma.from_texts(texts, OpenAIEmbeddings())
    retriever = db.as_retriever()

    ai = LangChainBasedAIChatParticipant(
        name="Assistant",
        chat_model=chat_model,
        # Pass the retriever to the AI participant
        retriever=retriever,
        spinner=spinner,
    )
    user = UserChatParticipant(name="User")
    participants = [user, ai]

    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(), renderer=TerminalChatRenderer(), initial_participants=participants
    )

    chat_conductor = RoundRobinChatConductor()
    chat_conductor.initiate_dialog(chat=chat)


if __name__ == "__main__":
    load_dotenv()

    typer.run(chatgpt_clone_with_langchain_retrieval)
