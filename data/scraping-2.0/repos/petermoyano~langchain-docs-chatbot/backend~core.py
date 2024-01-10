import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone

from consts import INDEX_NAME

# initialize pinecone client
pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
              environment=os.environ["PINECONE_ENVIRONMENT"])


def run_llm(query: str) -> any:
    embeddings = OpenAIEmbeddings()

    # instance of vector db
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings)

    chat = ChatOpenAI(verbose=True, temperature=0)

    # The RetrievalQA chain needs a retriever, which we can create by using the .as_retriever() method
    qa = RetrievalQA.from_chain_type(
        llm=chat, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

    return qa({"query": query})


if __name__ == '__main__':
    print(run_llm("What are the core modules of LangChain?"))
