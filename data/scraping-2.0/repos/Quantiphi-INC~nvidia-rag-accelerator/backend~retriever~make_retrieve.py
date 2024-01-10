# Takes in a query from the user and retrieves relevant context.
# Number of docs to return k=3

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Milvus
from pathlib import Path
from config import config_ns
import os
model_name = Path(Path(__file__).resolve().parents[1],f"embedding_model/{config_ns.retriever['emb_model_name']}").as_posix()
model_kwargs = {'device': config_ns.retriever['device_type']}
encode_kwargs = {'normalize_embeddings': True}
MILVUS_DB_HOST=os.getenv("MILVUS_DB_HOST")
# TODO: Create a singelton class for embedding_model
# used in two places: data_loader and retriever
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)



def get_context(query:str):
    retriever = Milvus(embedding_function=embedding_model,collection_name='LangChainCollection',drop_old=False,
                        connection_args={'host':MILVUS_DB_HOST})
    retriever = retriever.as_retriever(search_type='similarity')
    result = retriever.get_relevant_documents(query)
    result = [i.page_content for i in result]
    return result