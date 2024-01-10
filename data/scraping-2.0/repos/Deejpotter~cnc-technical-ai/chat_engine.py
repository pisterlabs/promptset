import os
import sys
import openai
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from chat_history import ChatHistory
from qa_manager import QAManager
from templates.system_prompt import system_prompt

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


class ChatEngine:
    """
    ChatEngine class handles the core functionality of the chat system.
    It uses OpenAI's language model for generating responses based on the input message and best practices fetched from the database.
    """

    def __init__(self):
        """
        Initializes the ChatEngine with necessary components and configurations.
        """
        # Set OpenAI API key
        try:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        except KeyError:
            sys.stderr.write("API key not found.")
            sys.exit(1)

        # Initialize ChatHistory for managing conversation history
        self.chat_history = ChatHistory()

        # Initialize DataManager for database interactions
        self.data_manager = QAManager()

        # Initialize the ChatOpenAI class and define the system prompt template
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

        # Use the system_prompt template to create a PromptTemplate instance
        self.system_template = system_prompt

        # Create a prompt template for the system prompt
        self.system_prompt = PromptTemplate(
            input_variables=["message", "best_practice"], template=self.system_template
        )
        self.chain = LLMChain(llm=llm, prompt=self.system_prompt)

    def process_user_input(self, message):
        """
        Processes the user input, retrieves best practices based on the input, and generates a bot response.
        Args:
            message (str): The user input message.
        Returns:
            str: The bot's response.
        """
        # Add user message to conversation history
        self.chat_history.add_message("user", message)

        # Generate best practices based on user input
        best_practices = self.generate_best_practice(message)

        # Generate bot response
        bot_response = self.chain.run(message=message, best_practice=best_practices)
        self.chat_history.add_message("bot", bot_response)

        return bot_response

    def generate_best_practice(self, user_message):
        """
        Generates best practices from the database based on the user message using vector search.
        Args:
            user_message (str): The user input message.
        Returns:
            List[str]: A list of best practices or similar responses.
        """
        try:
            query_vector = self.data_manager.create_vector_embeddings(user_message)
            similar_responses = self.data_manager.find(query_vector)

            # Extracting answers from the query response
            best_practices = [
                match["metadata"]["answer"] for match in similar_responses["matches"]
            ]

            return best_practices
        except Exception as e:
            print(f"Error in generating best practices: {e}")
            return []
