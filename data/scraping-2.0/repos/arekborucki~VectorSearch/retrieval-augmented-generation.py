import os
from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

# Retrieve environment variables for sensitive information
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

ATLAS_CONNECTION_STRING = os.getenv('ATLAS_CONNECTION_STRING')
if not ATLAS_CONNECTION_STRING:
    raise ValueError("The ATLAS_CONNECTION_STRING environment variable is not set.")

# Set the OPENAI_API_KEY in the environment
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

DB_NAME = "langchain"
COLLECTION_NAME = "vectorSearch"

def create_vector_search():
    """
    Creates a MongoDBAtlasVectorSearch object using the connection string, database, and collection names, along with the OpenAI embeddings and index configuration.
    """
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        ATLAS_CONNECTION_STRING,
        f"{DB_NAME}.{COLLECTION_NAME}",
        OpenAIEmbeddings(),
        index_name="default"
    )
    return vector_search

def perform_question_answering(query):
    """
    This function uses a retriever and a language model to answer a query based on the context from documents.
    """
    vector_search = create_vector_search()

    # Setup the vector search as a retriever for finding similar documents
    qa_retriever = vector_search.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 100, "post_filter_pipeline": [{"$limit": 1}]}
    )

    prompt_template = """
    Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}
    
    Question: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(max_tokens=100),
        chain_type="stuff",
        retriever=qa_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    docs = qa({"query": query})

    return docs["result"], docs['source_documents']

if __name__ == "__main__":
    # Example usage of the perform_question_answering function
    try:
        question = "Does MongoDB Atlas offer auditing?"
        answer, sources = perform_question_answering(question)
        print(f"Question: {question}")
        print("Answer:", answer)
        print("Source Documents:", sources)
    except Exception as e:
        print(f"An error occurred: {e}")
