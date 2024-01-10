import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

from utils import LOCAL_SOURCES_FILEPATH


EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
INDEX_NAME = os.environ["INDEX_NAME"]
TEXT_FIELD = "text"


def setup():
    """Initializes the local sources file, embedding model, vectorstore, and chat model."""
    if not os.path.exists(LOCAL_SOURCES_FILEPATH):
        with open(LOCAL_SOURCES_FILEPATH, 'w'):
            pass

    print("Initializing embedding model...")
    embed = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=OPENAI_API_KEY
    )

    print("Connecting to Pinecone...")
    index = pinecone.Index(INDEX_NAME)
    vectorstore = Pinecone(index, embed, TEXT_FIELD)

    print("Initializing chat model...")
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                     model_name="gpt-3.5-turbo", temperature=0.0)
    qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(
        search_kwargs=dict(k=3)), return_source_documents=True)

    return embed, index, vectorstore, qa
