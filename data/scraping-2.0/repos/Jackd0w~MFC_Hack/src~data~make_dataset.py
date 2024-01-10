from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredExcelLoader
from llama_index import download_loader
from pathlib import Path
from chromadb.utils import embedding_functions
import chromadb
from chromadb.config import Settings
from embedding import sentence_embeddings

# load the document and split it into chunks

# split it into chunks
chroma_client = chromadb.PersistentClient("src/data")

# create the open-source embedding function
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="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",
    model_name="ai-forever/sbert_large_nlu_ru"
)

collection = chroma_client.create_collection(name="embedding_vector", embedding_function=huggingface_ef)
# load it into Chroma
collection.add(embeddings=sentence_embeddings.tolist(), ids=[str(i) for i in range(0, 2733)])

# query it
query = "Какие основания для детской карты?"
#docs = collection.similarity_search(query)

# print results
print(collection)