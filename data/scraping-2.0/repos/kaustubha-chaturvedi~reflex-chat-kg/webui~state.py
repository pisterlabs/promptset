import os
from pyvis.network import Network
import networkx as nx
import reflex as rx
from dotenv import load_dotenv
from langchain.chains.conversation.memory import ConversationKGMemory
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
load_dotenv()


class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str


llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/"),
        max_tokens=50,
        temperature=0.7,
    )
template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
                If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.
                Relevant Information:
                {history}
                Conversation:
                Human: {input}
                AI:"""
"""The app state."""
prompt = PromptTemplate(input_variables=["history","input"],template=template)
conv=ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=ConversationKGMemory(llm=llm),
        verbose=True,
    )

class State(rx.State):
    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = {
        "Intros": [QA(question="What is your name?", answer="reflex")],
    }

    # The current chat name.
    current_chat = "Intros"

    # The currrent question.
    question: str

    # Whether we are processing the question.
    processing: bool = False

    # The name of the new chat.
    new_chat_name: str = ""

    # Whether the drawer is open.
    drawer_open: bool = False

    # Whether the modal is open.
    modal_open: bool = False

    def create_chat(self):
        """Create a new chat."""
        # Insert a default question.
        self.chats[self.new_chat_name] = [
            QA(question="What is your name?", answer="reflex")
        ]
        self.current_chat = self.new_chat_name

    def toggle_modal(self):
        """Toggle the new chat modal."""
        self.modal_open = not self.modal_open

    def toggle_drawer(self):
        """Toggle the drawer."""
        self.drawer_open = not self.drawer_open

    def delete_chat(self):
        """Delete the current chat."""
        del self.chats[self.current_chat]
        if len(self.chats) == 0:
            self.chats = {
                "New Chat": [QA(question="What is your name?", answer="reflex")]
            }
        self.current_chat = list(self.chats.keys())[0]
        self.toggle_drawer()

    def set_chat(self, chat_name: str):
        """Set the name of the current chat.

        Args:
            chat_name: The name of the chat.
        """
        self.current_chat = chat_name
        self.toggle_drawer()

    @rx.var
    def chat_titles(self) -> list[str]:
        """Get the list of chat titles.

        Returns:
            The list of chat names.
        """
        return list(self.chats.keys())

    async def process_question(self,form_data:dict[str,str]):
        
        # Check if we have already asked the last question or if the question is empty
        self.question = form_data["question"]
        if (self.chats[self.current_chat][-1].question == self.question or self.question == ""):
            return
        
        # Set the processing flag to true and yield.
        self.processing = True
        yield

        # Start a new session to answer the question.

        qa = QA(question=self.question, answer="")
        self.chats[self.current_chat].append(qa)
        self.chats[self.current_chat][-1].answer = conv.predict(input=self.question)
        self.chats = self.chats
        yield
        
        # Toggle the processing flag.
        self.processing = False
        plot_graph(conv.memory.kg.get_triples())

def plot_graph(chat):
    G = nx.Graph()
    G.add_node("reflex")
    for i in chat:
        G.add_node(i[0])
        G.add_node(i[2])
    for i in chat:
        G.add_edge("reflex",i[2])
        G.add_edge(i[0],i[2])
    nx.draw(G,with_labels=True)
    net = Network()
    net.from_nx(G)
    net.save_graph("assets/graph.html")