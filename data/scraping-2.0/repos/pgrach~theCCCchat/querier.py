import os
from dotenv import load_dotenv
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import langchain.vectorstores
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def initialize_environment():
    """Load environment variables and return necessary configurations."""
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
    namespace = os.getenv('NAMESPACE', default="default_namespace")
    return openai_api_key, pinecone_api_key, pinecone_env, namespace

def setup_search(pinecone_api_key, pinecone_env, namespace):
    """Initialize Pinecone and set up document search."""
    try:
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        embeddings = OpenAIEmbeddings()
        index_name = "langchain-demo"
        # Try to connect to an existing index
        index = langchain.vectorstores.Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)
        logging.info(f"Connected to existing index: {index_name}")
        return index
    except Exception as e:
        # Log the exception if connecting to the existing index fails
        logging.error(f"Failed to connect to existing index: {index_name}. Error: {e}")
        # Attempt to create a new index if it does not exist
        try:
            index = langchain.vectorstores.Pinecone(index_name=index_name, embeddings=embeddings, namespace=namespace)
            logging.info(f"Created new index: {index_name}")
            return index
        except Exception as e:
            # Log the exception if creating the index fails
            logging.error(f"Failed to create new index: {index_name}. Error: {e}")
            # Re-raise the exception to handle it upstream or terminate the script
            raise

def setup_llm(openai_api_key):
    """Set up the language model and the question-answering chain."""
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    template = """You are a chatbot having a conversation with a human.
Given the following extracted parts of a long document and a question, create a final answer.
{context}
{chat_history}
Human: {human_input}
Chatbot:"""
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    return load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt)

def user_interaction(docsearch, chain, namespace):
    """Interact with the user to get their queries and display results."""
    while True:
        user_query = input("Enter your question or type 'exit' to quit: ")
        if user_query.lower() == 'exit':
            break
        docs = docsearch.similarity_search(user_query, namespace=namespace)
        print(f"Retrieved Docs: {docs}")  # This will print the retrieved documents.
        answer = chain.run(input_documents=docs, question=user_query, human_input=user_query)
        print(f"Answer: {answer}")

def main():
    openai_api_key, pinecone_api_key, pinecone_env, namespace = initialize_environment()
    docsearch = setup_search(pinecone_api_key, pinecone_env, namespace)
    chain = setup_llm(openai_api_key)
    user_interaction(docsearch, chain, namespace)

# If the script is executed directly, call the main function
if __name__ == "__main__":
    main()