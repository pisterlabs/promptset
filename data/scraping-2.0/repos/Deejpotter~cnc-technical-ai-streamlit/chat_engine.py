import os
import sys
import openai
from chat_history import ChatHistory
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


class ChatEngine:
    # Initialize the ChatEngine object and its properties
    def __init__(self):
        # Set the OpenAI API key
        try:
            openai.api_key = os.environ['OPENAI_API_KEY']
        except KeyError:
            sys.stderr.write("API key not found.")
            exit(1)

        # Initialize ChatHistory
        self.chat_history = ChatHistory()

        # Clear existing conversation history and start with a clean slate
        self.chat_history.save_conversation_history([])

        # Load and embed the CSV data for best practices to create a vector store.
        loader = CSVLoader(file_path="MakerStoreTechnicalInfo.csv")
        documents = loader.load()
        embeddings = OpenAIEmbeddings()
        self.db = FAISS.from_documents(documents, embeddings)

        # Initialize the ChatOpenAI class and define the system prompt template
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
        self.system_template = """
        You Are Maker Bot.
        You are a customer service representative and sales assistant.
        You work for a company called Maker Store. Your job is to answer customer questions about the products and services offered by Maker Store.
        If someone asks if you sell a product, you should respond as if you sell all of the products that Maker Store sells.
        
        Help answer this question:
        {message}
        """
        # Create a prompt template for the system prompt
        self.system_prompt = PromptTemplate(
            input_variables=["message"],
            template=self.system_template
        )
        self.chain = LLMChain(llm=llm, prompt=self.system_prompt)

    # Method to handle user input and generate bot response
    def process_user_input(self, message):
        # Add the user message to the conversation history
        self.chat_history.add_message("user", message)

        # Search for the best practices
        best_practices = self.generate_best_practice(message)
        # Generate a bot response based on best practices and user message
        bot_response = self.generate_bot_response(message, best_practices)
        return bot_response

    # Method to generate a bot response based on best practices
    def generate_best_practice(self, user_message):
        # Search for similar responses in the database. The k parameter returns the top k most similar responses.
        similar_response = self.db.similarity_search(user_message, k=3)
        # Get the page content from the documents so we don't get the metadata.
        best_practice = [doc.page_content for doc in similar_response]
        return best_practice

    # Method to generate a bot response based on best practices and user message
    def generate_bot_response(self, message, best_practices):
        # Run the chain to get the best practices then generate the bot response
        bot_response = self.chain.run(message=message, best_practice=best_practices)
        # Add the bot response to the conversation history
        self.chat_history.add_message("bot", bot_response)
        return bot_response
