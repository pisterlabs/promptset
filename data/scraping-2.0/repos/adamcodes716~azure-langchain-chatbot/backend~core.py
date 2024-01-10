import os
from typing import Any, List, Dict
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import pinecone

# This retrieves from the vector store

# from consts import INDEX_NAME
INDEX_NAME = "langchain-doc-index"

# from dotenv import load_dotenv
load_dotenv()
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    #chat = ChatOpenAI(verbose=True, temperature=0)
    chat = AzureOpenAI(
                      openai_api_type="azure",
                      openai_api_key=os.getenv("AZURE_OPEN_API_KEY"),
                      openai_api_base=os.getenv("AZURE_OPENAI_BASE"),
                      deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
                      model=os.getenv("AZURE_DEPLOYMENT_NAME"),
                      temperature=0.7,
                      openai_api_version=os.getenv("OPENAI_VERSION"))       

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))