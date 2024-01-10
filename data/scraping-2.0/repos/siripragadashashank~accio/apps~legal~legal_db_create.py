import os
import uuid
import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from chromadb.config import Settings


from pathlib import Path
root = Path(__file__).parent.parent.parent
legal_data_path = os.path.join(root, 'data', 'legal')

loader = DirectoryLoader(legal_data_path, glob='./*.pdf', loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=5
)

docs = text_splitter.split_documents(documents)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

client = chromadb.HttpClient(
    host='localhost',
    port=8000,
    # settings=Settings(allow_reset=True)
)

# client.reset()
# the default embeddings are MiniLM-L6-v2, which is the same for Llama2
collection = client.create_collection("accio_legal_llama")

for doc in docs:
    collection.add(
        ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
    )

# db = Chroma(client=client,
#             collection_name="accio_legal",
#             embedding_function=embedding_function)
#
# query = "what is this legal agreement about"
# docs = db.similarity_search(query)
# print(docs[0].page_content)


