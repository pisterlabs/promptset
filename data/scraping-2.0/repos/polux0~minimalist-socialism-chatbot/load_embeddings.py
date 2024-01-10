from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
import openai
import os
from dotenv import load_dotenv
load_dotenv()

def load_embeddings():
    # access environment variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")

    openai.api_key = OPENAI_API_KEY

    # initialize embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load the vector database
    embeddings_persist_directory=f"./{DATA_DIRECTORY}-embeddings"

    vector_db = Chroma(persist_directory=embeddings_persist_directory, embedding_function=embedding_function)
    print(vector_db)
    
    return vector_db