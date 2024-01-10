from langchain.embeddings import OpenAIEmbeddings
import os
from langchain.vectorstores import Chroma
import dotenv

dotenv.load_dotenv()
name = "guofulun"
cache_dir = os.path.join(".embeddings/", name)
embedding = OpenAIEmbeddings()
vstore = Chroma(
    collection_name="my_collection",
    embedding_function=embedding,
    persist_directory=cache_dir,
)
import pdb

pdb.set_trace()
retriever = vstore.as_retriever()
