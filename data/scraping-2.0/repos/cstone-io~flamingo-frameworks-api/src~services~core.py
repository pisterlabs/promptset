import chromadb
from chromadb.config import Settings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from ..models.inputs import Query
from ..models.outputs import ChatResponse
from ..utils.config import Config

"""
Public Services
"""

config = Config.get()
chroma_kwargs = config.chromadb.to_dict()
chroma = chromadb.HttpClient(
    **chroma_kwargs, settings=Settings(chroma_api_impl="chromadb.api.fastapi.FastAPI")
)


async def chat(body: Query) -> ChatResponse:
    """
    Text report agent controller. This controller is responsible for handling
    requests to the /agents/text route.

    :param body: Query object
    :returns: string answer for successful requests
    """
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(client=chroma, embedding_function=embeddings)
    retriever = vector_db.as_retriever()

    llm_open = ChatOpenAI(
        model=config.langchain.model,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_open,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )

    chain_output = qa_chain(body.query)
    llm_response = ChatResponse(
        answer=chain_output["result"],
        sources=chain_output["source_documents"],
    )

    return llm_response
