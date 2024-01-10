# services/chat_service.py
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


class ChatService:
    def __init__(self, retriever):
        # Create a ChatOpenAI instance
        # Adjust the temperature parameter as needed
        self.model = ChatOpenAI(model='gpt-4', temperature=0)

        # Create a ConversationalRetrievalChain instance
        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            self.model, retriever=retriever)

    def ask(self, question, chat_history=None):
        # Initialize chat_history if it's not provided
        if chat_history is None:
            chat_history = []

        # Pass the question and chat_history to the ConversationalRetrievalChain instance
        result = self.conversational_chain(
            {"question": question, "chat_history": chat_history})

        # Append the question-answer pair to chat_history
        chat_history.append((question, result['answer']))

        # Return the answer and the updated chat_history
        return result['answer'], chat_history
