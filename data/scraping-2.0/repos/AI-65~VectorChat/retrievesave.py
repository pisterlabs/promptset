import os
import logging
import json
from dotenv import load_dotenv
# Importing necessary modules and classes from the 'langchain' package
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks import get_openai_callback
from typing import List
from pydantic import BaseModel, Field

# Setting up logging
logging.basicConfig(filename='logfilechoose.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Mapping Rechtsgebiet to specific vectorstore
VECTORSTORE_MAP = {
    '1': '134_vectorstore',
    '2': 'sachenrecht_vectorstore',
}

# Pydantic model for line list
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")

# Output parser class to parse lines of text
class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)
    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

# Decorator function to track OpenAI API usage
def track_openai_api(func):
    def wrapper(*args, **kwargs):
        with get_openai_callback() as cb:
            result = func(*args, **kwargs)
        logging.info(f"Total Tokens: {cb.total_tokens}")  # Logging the total tokens used
        logging.info(f"Prompt Tokens: {cb.prompt_tokens}")  # Logging the prompt tokens used
        logging.info(f"Completion Tokens: {cb.completion_tokens}")  # Logging the completion tokens used
        logging.info(f"Total Cost (USD): ${cb.total_cost}")  # Logging the total cost
        return result
    return wrapper

# Function to load vectorstore from disk
def load_vectorstore(save_path='faiss_index'):
    if os.path.exists(save_path):
        logging.info("Loading vectorstore from disk...")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(save_path, embeddings)
        logging.info("Loaded vectorstore from disk.")
        return vectorstore
    else:
        logging.error(f"Vectorstore not found at {save_path}. Please run create_embeddings.py first.")
        return None

# Function to setup language model chain
def setup_llm_chain():
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
        different versions in german of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    llm = ChatOpenAI(temperature=0)
    output_parser = LineListOutputParser()

    llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)

    return llm, llm_chain

# Function to create multi query retriever
@track_openai_api
def create_multi_query_retriever(vectorstore, llm, query):
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)
    retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)
    _filter = LLMChainFilter.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=_filter, base_retriever=retriever)
    filtered_docs = compression_retriever.get_relevant_documents(query)
    return filtered_docs

# Function to get user query
def get_query():
    question = input("Enter your question: ")
    logging.info("Running query: '%s'", question)
    return question

# Function to print retrieved documents
def print_results(filtered_docs):
    if filtered_docs:
        print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(filtered_docs)]))
    else:
        print("No results found for the given query.")

# Function to get user's choice for vectorstore
def get_vectorstore_choice():
    print("Select the Rechtsgebiet:")
    for key, value in VECTORSTORE_MAP.items():
        print(f"{key}: {value.replace('_vectorstore', '')}")
    choice = input("Enter the number corresponding to your choice: ")
    while choice not in VECTORSTORE_MAP:
        print("Invalid choice. Please enter a number from the list.")
        choice = input("Enter the number corresponding to your choice: ")
    return VECTORSTORE_MAP[choice]

# Function to save data to a JSON file
def save_data(document_context, user_input):
    data = {
        'document_context': [doc.page_content for doc in document_context],  # We save only the document content
        'query': user_input
    }

    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Main function to execute the program
def main():
    load_dotenv()

    # Get user's vectorstore choice
    vectorstore_choice = get_vectorstore_choice()

    # Load a vectorstore
    vectorstore = load_vectorstore(save_path=vectorstore_choice)

    if vectorstore:
        # Setup LLM Chain
        llm, llm_chain = setup_llm_chain()

        # Ask a question
        question = get_query()

        # Create MultiQueryRetriever and get relevant documents
        filtered_docs = create_multi_query_retriever(vectorstore, llm, question)

        # Save data
        save_data(filtered_docs, question)

        # Print results
        print_results(filtered_docs)

if __name__ == "__main__":
    main()
