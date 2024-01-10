import os
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import pinecone

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_API_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
index_name = "61b-chain"

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

docsearch = Pinecone.from_existing_index(index_name, embeddings, "website_data")

def askAssignment(query, namespace):
    docs = docsearch.similarity_search(query, include_metadata=True, namespace=namespace)
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        output = str(chain.run(input_documents=docs, question=query))
        print("API Call Meta Data: ")
        print(cb)
    return output