from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from env import QDRANT_URL, OPENAI_API_KEY, qdrant_Api_Key

client = QdrantClient(url=QDRANT_URL,api_key=qdrant_Api_Key)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Qdrant(client=client, collection_name="insurance", embeddings=embeddings)

matching_docs = vector_store.similarity_search(query="what is Domiciliary Hospitalization?",k=5)
for doc in matching_docs: 
  print(doc.page_content)
  print("-"*20)