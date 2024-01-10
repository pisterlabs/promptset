import os

import openai
import reflex as rx
from py2neo import Graph, Node, Relationship
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = os.getenv("OPENAI_API_BASE","https://api.openai.com/v1")
graph = Graph("neo4j://localhost:7687", auth=("neo4j", "Mohammad@90"))

class QA(rx.Base):
    """A question and answer pair."""

    question: str
    answer: str


class State(rx.State):
    """The app state."""

    # A dict from the chat name to the list of questions and answers.
    chats: dict[str, list[QA]] = {
        "Intros": [QA(question="What is your name?", answer="Mohammad")],
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
            QA(question="What is your name?", answer="Mohammad")
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
                "New Chat": [QA(question="What is your name?", answer="Mohammad")]
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

    async def process_question(self, form_data: dict[str, str]):
        """Get the response from the API.

        Args:
            form_data: A dict with the current question.
        """
        # Check if we have already asked the last question or if the question is empty
        self.question = form_data["question"]
        if (
            self.chats[self.current_chat][-1].question == self.question
            or self.question == ""
        ):
            return

        # Set the processing flag to true and yield.
        self.processing = True
        yield

        # Build the messages.
        messages = [
            { "role": "system", "content": "You are a friendly chatbot named Reflex."}
        ]

        for qa in self.chats[self.current_chat][1:]:
            messages.append({ "role": "user", "content": qa.question})
            messages.append({ "role": "assistant", "content": qa.answer})

        messages.append({ "role": "user", "content": self.question})
        
        relevant_context = self.get_relevant_context_from_graph(self.question)
        if relevant_context:
            messages.extend(relevant_context)

        # Start a new session to answer the question.
        session = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL","gpt-3.5-turbo"),
            messages=messages,
            # max_tokens=50,
            # n=1,
            stop=None,
            temperature=0.7,
            stream=True,  # Enable streaming
        )
        qa = QA(question=self.question, answer="")
        self.chats[self.current_chat].append(qa)

        # Stream the results, yielding after every word.
        for item in session:
            if hasattr(item.choices[0].delta, "content"):
                answer_text = item.choices[0].delta.content
                self.chats[self.current_chat][-1].answer += answer_text
                self.chats = self.chats
                
                user_node = Node("User", name="User")
                question_node = Node("Question", text=qa.question)
                answer_node = Node("Answer", text=qa.answer)
                asks_relation = Relationship(user_node, "ASKS", question_node)
                has_answer_relation = Relationship(question_node, "HAS_ANSWER", answer_node)

                graph.create(user_node)
                graph.create(question_node)
                graph.create(answer_node)
                graph.create(asks_relation)
                graph.create(has_answer_relation)
                yield

        # Toggle the processing flag.
        self.processing = False
        
    def get_relevant_context_from_graph(self, question: str) -> list:
        """Retrieve relevant context from the knowledge graph based on the current question."""
        relevant_context = []

        # Query the knowledge graph to find relevant context based on the question
        query = (
            f"MATCH (u:User)-[:ASKS]->(q:Question)-[:HAS_ANSWER]->(a:Answer) "
            f"WHERE q.text CONTAINS '{question}' "
            f"RETURN a.text AS answer"
        )

        result = graph.run(query)

        for record in result:
            answer = record["answer"]
            relevant_context.append({"role": "assistant", "content": answer})

        return relevant_context
