import os
import openai
import pinecone
from dotenv import load_dotenv

from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

######################################## QUERY DOCUMENTS TO GET ANSWER #################################################

def initialize_system():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

    pinecone.init(api_key = PINECONE_API_KEY,
                  environment = PINECONE_API_ENV)

    return print("Vector-Store Environment Initialized")


def answer_question_rag(index_name):
    llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0.0, max_tokens = 1000)
    chain = load_qa_chain(llm, chain_type = "stuff", verbose = True)
    docsearch = Pinecone.from_existing_index(index_name, OpenAIEmbeddings())

    query = input("Please enter your question here;")
    docs = docsearch.similarity_search(query)
    answer = chain.run(input_documents = docs, question = query)
    print(answer)

initialize_system()
answer_question_rag("rag-system")
