import os
import logging
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.critical("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit(1)

# Load the custom dataset from a specified directory
directory_path = "mydata/"
try:
    loader = DirectoryLoader(directory_path)
    documents = loader.load()
except Exception as e:
    logger.critical(f"Failed to load documents from {directory_path}: {e}")
    exit(1)

# Split the documents into chunks for processing
chunk_size = 1000
chunk_overlap = 0
text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
texts = text_splitter.split_documents(documents)

# Embed the documents using OpenAI Embeddings and store them in a vector database
try:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = Chroma.from_documents(texts, embeddings)
except Exception as e:
    logger.critical(f"Failed to create embeddings: {e}")
    exit(1)

# Initialize the language model from OpenAI
try:
    llm = OpenAI(openai_api_key=openai_api_key)
except Exception as e:
    logger.critical(f"Failed to initialize the language model: {e}")
    exit(1)

# Create the ConversationalRetrievalChain with the necessary configurations
try:
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
    )
except Exception as e:
    logger.critical(f"Failed to create ConversationalRetrievalChain: {e}")
    exit(1)

# Function to handle conversational interactions
def converse(chain, max_history=5):
    chat_history = []

    while True:
        query = input("Prompt: ").strip()
        
        if query.lower() in ["quit", "q", "exit"]:
            logger.info("Exiting the conversational loop.")
            break

        # Call the chain with the user's query and chat history
        try:
            result = chain({"question": query, "chat_history": chat_history})
            logger.info("Retrieved answer successfully.")
            print(result["answer"])
        except Exception as e:
            logger.error(f"Failed to retrieve answer: {e}")
            continue

        # Append to chat history and maintain its size
        chat_history.append((query, result["answer"]))
        if len(chat_history) > max_history:
            chat_history.pop(0)

# Main entry point for the application
if __name__ == "__main__":
    logger.info("Starting conversational interface.")
    converse(chain)
