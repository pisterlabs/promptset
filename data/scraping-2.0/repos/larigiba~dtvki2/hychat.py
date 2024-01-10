import os
import torch
import pinecone

from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers import PineconeHybridSearchRetriever

device = None
OPENAI_API_KEY = None


def init_env():
    global OPENAI_API_KEY
    global device

    OPENAI_API_KEY = "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"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # initialize pinecone
    pinecone.init(
        api_key="a212522e-6af7-4e80-bb48-5ac2d3532842",  # "54ceda71-d9a7-44aa-b5e7-1cc49f403009",  # find at app.pinecone.io
        environment="gcp-starter",  # next to api key in console
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomRetriever(BaseRetriever):
    vectorstore: PineconeHybridSearchRetriever

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ):
        """
        _get_relevant_documents is function of BaseRetriever implemented here

        :param query: String value of the query

        """
        results = self.vectorstore.get_relevant_documents(query=query)
        return [doc for doc in results if doc.metadata["document_type"] != "Neuerungen"]


def get_chain():
    # get dense model
    model_name = "text-embedding-ada-002"
    embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

    # get sparse model
    sparse_model_id = "naver/splade-cocondenser-ensembledistil"
    sparse_model = Splade(sparse_model_id, agg="max")
    sparse_model.to(device)  # move to GPU if possible
    sparse_model.eval()

    # get vectorstore inded
    index_name = "dakip"
    index = pinecone.Index(index_name)
    index.describe_index_stats()

    # get llm
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, model_name="gpt-4-1106-preview", temperature=0.8
    )

    # LangChain Retriever
    hybridretriever = PineconeHybridSearchRetriever(
        embeddings=embed, sparse_encoder=sparse_model, index=index
    )

    # Add filtering via CustomRetriever
    filtered_retriever = CustomRetriever(vectorstore=hybridretriever)

    # LangChain QA
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff", retriever=filtered_retriever
    )

    return qa_with_sources
