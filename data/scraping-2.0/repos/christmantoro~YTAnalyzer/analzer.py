# Step-1: Import necessary libraries

from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPEN_API_KEY"] = os.environ.get('OPENAI_API_KEY')

# Initialize OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Initialize Chroma vector store for document search
docsearch = Chroma(persist_directory="db", embedding_function=embeddings)

# Create a RetrievalQAWithSourcesChain using OpenAI model and Chroma vector store
chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0),
                                                    chain_type="stuff",
                                                    retriver=docsearch.as_retriever(search_kwargs={"k":1}))

def get_analysis(user_input):
    """
    Get the analysis for a given user input.

    Args:
        user_input (str): The user input/question.

    Returns:
        str: The answer/result of the analysis.
    """
    # Get the result from the chain and return only the outputs
    result = chain({"question": user_input}, return_only_outputs=True)
    return result["answer"]